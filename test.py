"""
Adaptive Quiz / Intelligent Tutoring System - Supabase Integration
===================================================================
Architecture: BKT (mastery tracking) + FSRS (scheduling) + UCB (topic selection)

Database: Supabase (PostgreSQL)
- Users table
- Topics table
- User_Topics table (BKT + FSRS state)
- Review_History table
"""

import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import random

# Supabase client
try:
    from supabase import create_client, Client
except ImportError:
    print("Install supabase: pip install supabase")
    raise


# ==========================================================================================
# CONFIGURATION
# ==========================================================================================

class SupabaseConfig:
    """Supabase connection configuration"""

    # TODO: Replace with your actual Supabase credentials
    SUPABASE_URL = "https://your-project.supabase.co"
    SUPABASE_KEY = "your-anon-key"

    @classmethod
    def get_client(cls) -> Client:
        """Get Supabase client instance"""
        return create_client(cls.SUPABASE_URL, cls.SUPABASE_KEY)


# ==========================================================================================
# DATA MODELS
# ==========================================================================================

class Rating(Enum):
    """
    User performance rating - maps to quality scores

    Quality Score Mapping:
    0 = Complete blackout → AGAIN
    1 = Incorrect, but familiar → AGAIN
    2 = Incorrect, but almost got it → HARD
    3 = Correct with difficulty → HARD
    4 = Correct after hesitation → GOOD
    5 = Perfect recall → EASY
    """
    AGAIN = 1  # Quality 0-1: Wrong
    HARD = 2  # Quality 2-3: Correct but struggled
    GOOD = 3  # Quality 4: Correct, normal
    EASY = 4  # Quality 5: Perfect

    @classmethod
    def from_quality_score(cls, quality: int) -> 'Rating':
        """
        Convert quality score (0-5) to Rating enum.

        Args:
            quality: Integer from 0 (complete failure) to 5 (perfect)

        Returns:
            Rating enum value
        """
        if quality <= 1:
            return cls.AGAIN
        elif quality <= 3:
            return cls.HARD
        elif quality == 4:
            return cls.GOOD
        else:  # quality == 5
            return cls.EASY


@dataclass
class BKTParameters:
    """Bayesian Knowledge Tracing state"""
    p_init: float = 0.1
    p_learn: float = 0.3
    p_slip: float = 0.1
    p_guess: float = 0.25
    p_mastery: float = 0.1


@dataclass
class FSRSCard:
    """FSRS scheduling state"""
    stability: float = 0.0
    difficulty: float = 5.0
    reps: int = 0
    lapses: int = 0
    last_review: Optional[str] = None
    next_review_minutes: float = 0.0


@dataclass
class Topic:
    """Topic with learning state"""
    id: str
    name: str
    description: Optional[str]
    bkt: BKTParameters
    fsrs: FSRSCard
    times_selected: int = 0


# ==========================================================================================
# BKT ENGINE
# ==========================================================================================

class BKTEngine:
    """Bayesian Knowledge Tracing"""

    @staticmethod
    def update(bkt: BKTParameters, correct: bool) -> float:
        """Update mastery probability"""
        p_know = bkt.p_mastery

        if correct:
            p_correct_know = 1 - bkt.p_slip
            p_correct_dont = bkt.p_guess

            numerator = p_correct_know * p_know
            denominator = p_correct_know * p_know + p_correct_dont * (1 - p_know)

            p_know_after = numerator / denominator if denominator > 0 else p_know
        else:
            p_wrong_know = bkt.p_slip
            p_wrong_dont = 1 - bkt.p_guess

            numerator = p_wrong_know * p_know
            denominator = p_wrong_know * p_know + p_wrong_dont * (1 - p_know)

            p_know_after = numerator / denominator if denominator > 0 else p_know

        bkt.p_mastery = p_know_after + (1 - p_know_after) * bkt.p_learn
        return bkt.p_mastery

    @staticmethod
    def is_mastered(bkt: BKTParameters, threshold: float = 0.90) -> bool:
        """Check if mastered"""
        return bkt.p_mastery >= threshold


# ==========================================================================================
# FSRS ENGINE
# ==========================================================================================

class FSRSEngine:
    """Free Spaced Repetition Scheduler"""

    INIT_STABILITY = {
        Rating.AGAIN: 0.003,
        Rating.HARD: 0.01,
        Rating.GOOD: 0.02,
        Rating.EASY: 0.05
    }

    @staticmethod
    def schedule(card: FSRSCard, rating: Rating, timestamp: datetime) -> FSRSCard:
        """Calculate next review interval"""

        new_card = FSRSCard(
            stability=card.stability,
            difficulty=card.difficulty,
            reps=card.reps + 1,
            lapses=card.lapses,
            last_review=timestamp.isoformat()
        )

        if card.reps == 0:
            new_card.stability = FSRSEngine.INIT_STABILITY[rating]
            new_card.difficulty = 5.0
        else:
            if rating == Rating.AGAIN:
                new_card.stability = card.stability * 0.2
                new_card.difficulty = min(10.0, card.difficulty + 2.0)
                new_card.lapses += 1
            else:
                multipliers = {
                    Rating.HARD: 1.5,
                    Rating.GOOD: 2.5,
                    Rating.EASY: 4.0
                }

                difficulty_factor = 1.0 - (card.difficulty / 20.0)
                multiplier = multipliers[rating] * (0.7 + 0.6 * difficulty_factor)

                new_card.stability = card.stability * multiplier

                diff_changes = {
                    Rating.HARD: 1.0,
                    Rating.GOOD: 0.0,
                    Rating.EASY: -1.0
                }
                new_card.difficulty = max(1.0, min(10.0, card.difficulty + diff_changes[rating]))

        new_card.next_review_minutes = new_card.stability * 24 * 60

        return new_card

    @staticmethod
    def is_due(card: FSRSCard, current_time: datetime) -> bool:
        """Check if due for review"""
        if card.last_review is None:
            return True

        last_time = datetime.fromisoformat(card.last_review)
        minutes_since = (current_time - last_time).total_seconds() / 60.0

        return minutes_since >= card.next_review_minutes


# ==========================================================================================
# UCB SELECTOR
# ==========================================================================================

class UCBSelector:
    """Upper Confidence Bound topic selector"""

    @staticmethod
    def select(topics: List[Topic], total_questions: int, c: float = 1.4) -> Topic:
        """Select best topic using UCB1"""
        if not topics:
            raise ValueError("No topics available")

        for topic in topics:
            if topic.times_selected == 0:
                return topic

        best_topic = None
        best_score = float('-inf')

        for topic in topics:
            exploitation = 1.0 - topic.bkt.p_mastery
            exploration = c * math.sqrt(math.log(total_questions + 1) / topic.times_selected)
            score = exploitation + exploration

            if score > best_score:
                best_score = score
                best_topic = topic

        return best_topic


# ==========================================================================================
# SUPABASE DATA ACCESS LAYER
# ==========================================================================================

class SupabaseDAO:
    """Data Access Object for Supabase operations"""

    def __init__(self, client: Client):
        self.client = client

    # ==================================================================================
    # USER OPERATIONS
    # ==================================================================================

    def create_user(self, username: str, email: Optional[str] = None) -> str:
        """
        Create new user.

        Returns:
            user_id (UUID as string)
        """
        result = self.client.table('users').insert({
            'username': username,
            'email': email
        }).execute()

        return result.data[0]['id']

    def get_user(self, user_id: str) -> Optional[Dict]:
        """Get user by ID"""
        result = self.client.table('users').select('*').eq('id', user_id).execute()
        return result.data[0] if result.data else None

    def get_user_by_username(self, username: str) -> Optional[Dict]:
        """Get user by username"""
        result = self.client.table('users').select('*').eq('username', username).execute()
        return result.data[0] if result.data else None

    def update_user_activity(self, user_id: str, total_questions: int):
        """Update user's last active time and question count"""
        self.client.table('users').update({
            'last_active': datetime.now().isoformat(),
            'total_questions': total_questions
        }).eq('id', user_id).execute()

    # ==================================================================================
    # TOPIC OPERATIONS
    # ==================================================================================

    def create_topic(self, name: str, description: Optional[str] = None) -> str:
        """
        Create new topic.

        Returns:
            topic_id (UUID as string)
        """
        result = self.client.table('topics').insert({
            'name': name,
            'description': description
        }).execute()

        return result.data[0]['id']

    def get_all_topics(self) -> List[Dict]:
        """Get all topics"""
        result = self.client.table('topics').select('*').execute()
        return result.data

    def get_topic(self, topic_id: str) -> Optional[Dict]:
        """Get topic by ID"""
        result = self.client.table('topics').select('*').eq('id', topic_id).execute()
        return result.data[0] if result.data else None

    # ==================================================================================
    # USER_TOPIC OPERATIONS (BKT + FSRS State)
    # ==================================================================================

    def create_user_topic(self, user_id: str, topic_id: str) -> str:
        """
        Initialize user-topic relationship with default BKT/FSRS values.

        Returns:
            user_topic_id (UUID as string)
        """
        result = self.client.table('user_topics').insert({
            'user_id': user_id,
            'topic_id': topic_id
        }).execute()

        return result.data[0]['id']

    def get_user_topics(self, user_id: str) -> List[Dict]:
        """
        Get all topics for a user with their learning state.

        Returns:
            List of user_topic records with topic details joined
        """
        result = self.client.table('user_topics').select(
            '*, topics(*)'
        ).eq('user_id', user_id).execute()

        return result.data

    def get_user_topic(self, user_id: str, topic_id: str) -> Optional[Dict]:
        """Get specific user-topic state"""
        result = self.client.table('user_topics').select(
            '*'
        ).eq('user_id', user_id).eq('topic_id', topic_id).execute()

        return result.data[0] if result.data else None

    def update_user_topic_state(self, user_id: str, topic_id: str,
                                bkt: BKTParameters, fsrs: FSRSCard,
                                times_selected: int):
        """Update BKT and FSRS state for user-topic"""
        self.client.table('user_topics').update({
            # BKT
            'bkt_p_mastery': bkt.p_mastery,
            'bkt_p_init': bkt.p_init,
            'bkt_p_learn': bkt.p_learn,
            'bkt_p_slip': bkt.p_slip,
            'bkt_p_guess': bkt.p_guess,

            # FSRS
            'fsrs_stability': fsrs.stability,
            'fsrs_difficulty': fsrs.difficulty,
            'fsrs_reps': fsrs.reps,
            'fsrs_lapses': fsrs.lapses,
            'fsrs_last_review': fsrs.last_review,
            'fsrs_next_review_minutes': fsrs.next_review_minutes,

            # Metadata
            'times_selected': times_selected
        }).eq('user_id', user_id).eq('topic_id', topic_id).execute()

    # ==================================================================================
    # REVIEW HISTORY OPERATIONS
    # ==================================================================================

    def add_review(self, user_id: str, topic_id: str, correct: bool,
                   quality_score: int, rating: Rating, mastery_before: float,
                   mastery_after: float, timestamp: datetime,
                   response_time_ms: Optional[int] = None):
        """Record a review in history"""
        self.client.table('review_history').insert({
            'user_id': user_id,
            'topic_id': topic_id,
            'timestamp': timestamp.isoformat(),
            'correct': correct,
            'quality_score': quality_score,
            'rating': rating.name,
            'mastery_before': mastery_before,
            'mastery_after': mastery_after,
            'response_time_ms': response_time_ms
        }).execute()

    def get_user_review_history(self, user_id: str, limit: int = 100) -> List[Dict]:
        """Get recent review history for user"""
        result = self.client.table('review_history').select(
            '*'
        ).eq('user_id', user_id).order('timestamp', desc=True).limit(limit).execute()

        return result.data

    def get_topic_review_history(self, user_id: str, topic_id: str) -> List[Dict]:
        """Get review history for specific topic"""
        result = self.client.table('review_history').select(
            '*'
        ).eq('user_id', user_id).eq('topic_id', topic_id).order('timestamp').execute()

        return result.data


# ==========================================================================================
# MAIN ADAPTIVE QUIZ SYSTEM (SUPABASE VERSION)
# ==========================================================================================

class AdaptiveQuizSupabase:
    """Adaptive Quiz System with Supabase backend"""

    def __init__(self, supabase_client: Client, user_id: str):
        self.dao = SupabaseDAO(supabase_client)
        self.user_id = user_id
        self.topics: Dict[str, Topic] = {}
        self.total_questions = 0
        self.session_start = None

        self.bkt = BKTEngine()
        self.fsrs = FSRSEngine()
        self.ucb = UCBSelector()

        # Load user data
        self._load_user_data()

    def _load_user_data(self):
        """Load user and their topic states from database"""
        user = self.dao.get_user(self.user_id)
        if not user:
            raise ValueError(f"User {self.user_id} not found")

        self.total_questions = user['total_questions']

        # Load topics with state
        user_topics = self.dao.get_user_topics(self.user_id)

        for ut in user_topics:
            topic_data = ut['topics']

            self.topics[topic_data['id']] = Topic(
                id=topic_data['id'],
                name=topic_data['name'],
                description=topic_data.get('description'),
                bkt=BKTParameters(
                    p_init=ut['bkt_p_init'],
                    p_learn=ut['bkt_p_learn'],
                    p_slip=ut['bkt_p_slip'],
                    p_guess=ut['bkt_p_guess'],
                    p_mastery=ut['bkt_p_mastery']
                ),
                fsrs=FSRSCard(
                    stability=ut['fsrs_stability'],
                    difficulty=ut['fsrs_difficulty'],
                    reps=ut['fsrs_reps'],
                    lapses=ut['fsrs_lapses'],
                    last_review=ut['fsrs_last_review'],
                    next_review_minutes=ut['fsrs_next_review_minutes']
                ),
                times_selected=ut['times_selected']
            )

    def start_session(self):
        """Mark session start"""
        self.session_start = datetime.now()

    def get_next_topic(self, current_time: datetime = None) -> Optional[Topic]:
        """Select next topic using UCB"""
        if current_time is None:
            current_time = datetime.now()

        active = []
        for topic in self.topics.values():
            if not self.bkt.is_mastered(topic.bkt):
                if self.fsrs.is_due(topic.fsrs, current_time):
                    active.append(topic)

        if not active:
            return None

        return self.ucb.select(active, self.total_questions)

    def submit_answer(self, topic_id: str, correct: bool, quality_score: int,
                      timestamp: datetime = None, response_time_ms: Optional[int] = None) -> Dict:
        """
        Process answer and update database.

        Args:
            topic_id: Topic UUID
            correct: Whether answer was correct
            quality_score: Quality score 0-5 (0=blackout, 5=perfect)
            timestamp: When answered (defaults to now)
            response_time_ms: How long it took in milliseconds

        Returns:
            Result dictionary with mastery changes
        """
        topic = self.topics[topic_id]

        if timestamp is None:
            timestamp = datetime.now()

        # Convert quality score to rating
        rating = Rating.from_quality_score(quality_score)

        # Update BKT
        old_mastery = topic.bkt.p_mastery
        new_mastery = self.bkt.update(topic.bkt, correct)

        # Update FSRS
        topic.fsrs = self.fsrs.schedule(topic.fsrs, rating, timestamp)

        # Update counts
        topic.times_selected += 1
        self.total_questions += 1

        # Save to database
        self.dao.update_user_topic_state(
            self.user_id, topic_id, topic.bkt, topic.fsrs, topic.times_selected
        )

        self.dao.add_review(
            self.user_id, topic_id, correct, quality_score, rating,
            old_mastery, new_mastery, timestamp, response_time_ms
        )

        self.dao.update_user_activity(self.user_id, self.total_questions)

        return {
            'topic_name': topic.name,
            'mastery_before': old_mastery,
            'mastery_after': new_mastery,
            'mastery_change': new_mastery - old_mastery,
            'is_mastered': self.bkt.is_mastered(topic.bkt),
            'next_review_minutes': topic.fsrs.next_review_minutes,
            'total_reviews': topic.fsrs.reps,
            'rating': rating.name
        }

    def get_summary(self) -> Dict:
        """Get session summary"""
        mastered = sum(1 for t in self.topics.values() if self.bkt.is_mastered(t.bkt))
        avg_mastery = sum(t.bkt.p_mastery for t in self.topics.values()) / len(self.topics) if self.topics else 0

        duration_min = 0.0
        if self.session_start:
            duration_min = (datetime.now() - self.session_start).total_seconds() / 60.0

        return {
            'total_questions': self.total_questions,
            'topics_total': len(self.topics),
            'topics_mastered': mastered,
            'average_mastery': avg_mastery,
            'duration_minutes': duration_min
        }

    def get_review_history(self, limit: int = 50) -> List[Dict]:
        """Get recent review history"""
        return self.dao.get_user_review_history(self.user_id, limit)


# ==========================================================================================
# HELPER FUNCTIONS
# ==========================================================================================

def setup_user_with_topics(client: Client, username: str, topic_names: List[str]) -> str:
    """
    Helper to create a new user and initialize their topics.

    Args:
        client: Supabase client
        username: Username for new user
        topic_names: List of topic names to add

    Returns:
        user_id (UUID as string)
    """
    dao = SupabaseDAO(client)

    # Create user
    user_id = dao.create_user(username)

    # Get or create topics
    existing_topics = {t['name']: t['id'] for t in dao.get_all_topics()}

    for topic_name in topic_names:
        if topic_name in existing_topics:
            topic_id = existing_topics[topic_name]
        else:
            topic_id = dao.create_topic(topic_name)

        # Initialize user-topic relationship
        dao.create_user_topic(user_id, topic_id)

    return user_id


# ==========================================================================================
# EXAMPLE USAGE
# ==========================================================================================

def example_usage():
    """
    Example of how to use the Supabase-integrated system.

    NOTE: This is a template - you'll need to:
    1. Set up your Supabase project
    2. Run the SQL schema above
    3. Update SupabaseConfig with your credentials
    """

    print("=" * 80)
    print(" ADAPTIVE QUIZ SYSTEM - Supabase Integration Example")
    print("=" * 80)
    print()

    # Initialize Supabase client
    client = SupabaseConfig.get_client()

    # Setup new user with topics
    print("Creating new user with topics...")
    user_id = setup_user_with_topics(
        client,
        username="student_demo",
        topic_names=[
            "Variables & Data Types",
            "Control Flow (if/else)",
            "Loops (for/while)",
            "Functions & Parameters"
        ]
    )
    print(f"✓ Created user: {user_id}")
    print()

    # Initialize quiz system
    quiz = AdaptiveQuizSupabase(client, user_id)
    quiz.start_session()

    print(f"📚 Loaded {len(quiz.topics)} topics")
    print()

    # Simulate some questions
    print("Simulating quiz session...")
    print("-" * 80)

    for i in range(10):
        topic = quiz.get_next_topic()

        if not topic:
            print("All topics mastered!")
            break

        # Simulate answer (random for demo)
        correct = random.random() > 0.3
        quality_score = random.randint(3, 5) if correct else random.randint(0, 2)

        result = quiz.submit_answer(
            topic_id=topic.id,
            correct=correct,
            quality_score=quality_score
        )

        status = "✓" if correct else "✗"
        print(f"Q{i + 1}: {status} {topic.name[:30]:<30} | "
              f"Quality: {quality_score} ({result['rating']}) | "
              f"Mastery: {result['mastery_after']:.1%} ({result['mastery_change']:+.2%})")

    print()
    print("-" * 80)

    # Summary
    summary = quiz.get_summary()
    print("\nSESSION SUMMARY")
    print("=" * 80)
    print(f"Questions answered:     {summary['total_questions']}")
    print(f"Topics mastered:        {summary['topics_mastered']}/{summary['topics_total']}")
    print(f"Average mastery:        {summary['average_mastery']:.1%}")
    print()
    print("✓ All data persisted to Supabase!")
    print()

    print("=" * 80)
    print("Next steps:")
    print("1. Set up Supabase project at https://supabase.com")
    print("2. Run the SQL schema to create tables")
    print("3. Update SupabaseConfig with your credentials")
    print("4. Install: pip install supabase")
    print("=" * 80)


if __name__ == "__main__":
    print("""
    ⚠️  SETUP REQUIRED

    Before running this example:
    1. Create a Supabase project at https://supabase.com
    2. Copy the SQL schema (SUPABASE_SCHEMA) and run it in your SQL editor
    3. Update SupabaseConfig with your URL and anon key
    4. Install: pip install supabase

    Then uncomment the line below to run the example.
    """)

    # example_usage()  # Uncomment after setup
