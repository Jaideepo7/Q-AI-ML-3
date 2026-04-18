"""
Adaptive Quiz / Intelligent Tutoring System - Production Version
=================================================================
Architecture: BKT (mastery tracking) + FSRS (scheduling) + UCB (topic selection)

Database: Supabase (PostgreSQL)
- Users table
- Topics table
- User_Topics table (BKT + FSRS state)
- Review_History table

Author: Adaptive Learning System
Version: 2.0 - Production Ready
"""

import os
import math
import json
import base64
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==========================================================================================
# CONFIGURATION
# ==========================================================================================
try:
    from supabase import create_client, Client
except ImportError:
    logger.error("Supabase library not installed. Run: pip install supabase")
    raise ImportError("Please install supabase: pip install supabase")


class SupabaseConfig:
    """
    Simple Supabase Configuration
    """
    SUPABASE_URL: Optional[str] = None
    SUPABASE_KEY: Optional[str] = None

    @classmethod
    def get_url(cls) -> str:
        url = cls.SUPABASE_URL or os.getenv('SUPABASE_URL')
        if not url:
            raise ValueError("SUPABASE_URL is not set! Please configure it.")
        return url.rstrip('/')

    @classmethod
    def get_key(cls) -> str:
        key = cls.SUPABASE_KEY or os.getenv('SUPABASE_KEY')
        if not key:
            raise ValueError("SUPABASE_KEY is not set! Please configure it.")
        return key

    @classmethod
    def get_client(cls) -> Client:
        """Create and return Supabase client"""
        url = cls.get_url()
        key = cls.get_key()

        logger.info(f"Connecting to Supabase: {url}")

        try:
            client: Client = create_client(url, key)
            logger.info("✅ Supabase client connected successfully")
            return client
        except Exception as e:
            logger.error(f"Failed to connect to Supabase: {e}")
            raise
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
    AGAIN = 1
    HARD = 2
    GOOD = 3
    EASY = 4

    @classmethod
    def from_quality_score(cls, quality: int) -> 'Rating':
        """Convert quality score (0-5) to Rating enum"""
        if quality <= 1:
            return cls.AGAIN
        elif quality <= 3:
            return cls.HARD
        elif quality == 4:
            return cls.GOOD
        else:
            return cls.EASY


@dataclass
class BKTParameters:
    """Bayesian Knowledge Tracing state"""
    p_init: float = 0.1      # Initial knowledge probability
    p_learn: float = 0.3     # Learning rate per question
    p_slip: float = 0.1      # Probability of incorrect despite knowing
    p_guess: float = 0.25    # Probability of correct guess
    p_mastery: float = 0.1   # Current mastery probability

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)


@dataclass
class FSRSCard:
    """FSRS (Free Spaced Repetition Scheduler) card state"""
    stability: float = 0.0                    # Memory stability in days
    difficulty: float = 5.0                   # Item difficulty (1-10)
    reps: int = 0                             # Number of reviews
    lapses: int = 0                           # Number of times forgotten
    last_review: Optional[str] = None         # ISO timestamp of last review
    next_review_minutes: float = 0.0          # Minutes until next review

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)


@dataclass
class Topic:
    """Topic with learning state"""
    id: str
    name: str
    description: Optional[str]
    bkt: BKTParameters
    fsrs: FSRSCard
    times_selected: int = 0
    created_at: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'bkt': self.bkt.to_dict(),
            'fsrs': self.fsrs.to_dict(),
            'times_selected': self.times_selected,
            'created_at': self.created_at
        }


# ==========================================================================================
# BKT ENGINE
# ==========================================================================================

class BKTEngine:
    """
    Bayesian Knowledge Tracing Engine

    Tracks student mastery probability using Bayesian inference.
    """

    @staticmethod
    def update(bkt: BKTParameters, correct: bool) -> float:
        """
        Update mastery probability based on student response.

        Args:
            bkt: Current BKT parameters
            correct: Whether student answered correctly

        Returns:
            Updated mastery probability
        """
        p_know = bkt.p_mastery

        if correct:
            # Student got it right
            p_correct_know = 1 - bkt.p_slip
            p_correct_dont = bkt.p_guess

            numerator = p_correct_know * p_know
            denominator = p_correct_know * p_know + p_correct_dont * (1 - p_know)

            p_know_after = numerator / denominator if denominator > 0 else p_know
        else:
            # Student got it wrong
            p_wrong_know = bkt.p_slip
            p_wrong_dont = 1 - bkt.p_guess

            numerator = p_wrong_know * p_know
            denominator = p_wrong_know * p_know + p_wrong_dont * (1 - p_know)

            p_know_after = numerator / denominator if denominator > 0 else p_know

        # Apply learning
        bkt.p_mastery = p_know_after + (1 - p_know_after) * bkt.p_learn

        return bkt.p_mastery

    @staticmethod
    def is_mastered(bkt: BKTParameters, threshold: float = 0.90) -> bool:
        """Check if topic is mastered"""
        return bkt.p_mastery >= threshold


# ==========================================================================================
# FSRS ENGINE
# ==========================================================================================

class FSRSEngine:
    """
    Free Spaced Repetition Scheduler

    Optimizes review intervals based on memory science.
    """

    # Initial stability values for first review
    INIT_STABILITY = {
        Rating.AGAIN: 0.003,  # ~4 minutes
        Rating.HARD: 0.01,    # ~14 minutes
        Rating.GOOD: 0.02,    # ~29 minutes
        Rating.EASY: 0.05     # ~72 minutes
    }

    @staticmethod
    def schedule(card: FSRSCard, rating: Rating, timestamp: datetime) -> FSRSCard:
        """
        Calculate next review interval based on rating.

        Args:
            card: Current card state
            rating: Student's performance rating
            timestamp: When the review occurred

        Returns:
            Updated card state with new review time
        """
        new_card = FSRSCard(
            stability=card.stability,
            difficulty=card.difficulty,
            reps=card.reps + 1,
            lapses=card.lapses,
            last_review=timestamp.isoformat()
        )

        if card.reps == 0:
            # First review - use initial stability
            new_card.stability = FSRSEngine.INIT_STABILITY[rating]
            new_card.difficulty = 5.0
        else:
            # Subsequent reviews
            if rating == Rating.AGAIN:
                # Failed review - reduce stability
                new_card.stability = card.stability * 0.2
                new_card.difficulty = min(10.0, card.difficulty + 2.0)
                new_card.lapses += 1
            else:
                # Successful review - increase stability
                multipliers = {
                    Rating.HARD: 1.5,
                    Rating.GOOD: 2.5,
                    Rating.EASY: 4.0
                }

                # Adjust multiplier based on difficulty
                difficulty_factor = 1.0 - (card.difficulty / 20.0)
                multiplier = multipliers[rating] * (0.7 + 0.6 * difficulty_factor)

                new_card.stability = card.stability * multiplier

                # Update difficulty
                diff_changes = {
                    Rating.HARD: 1.0,   # Harder
                    Rating.GOOD: 0.0,   # Same
                    Rating.EASY: -1.0   # Easier
                }
                new_card.difficulty = max(1.0, min(10.0, card.difficulty + diff_changes[rating]))

        # Convert stability (days) to next review time (minutes)
        new_card.next_review_minutes = new_card.stability * 24 * 60

        return new_card

    @staticmethod
    def is_due(card: FSRSCard, current_time: datetime) -> bool:
        """
        Check if card is due for review.

        Args:
            card: Card to check
            current_time: Current timestamp

        Returns:
            True if review is due
        """
        if card.last_review is None:
            return True

        last_time = datetime.fromisoformat(card.last_review)
        minutes_since = (current_time - last_time).total_seconds() / 60.0

        return minutes_since >= card.next_review_minutes


# ==========================================================================================
# UCB SELECTOR
# ==========================================================================================

class UCBSelector:
    """
    Upper Confidence Bound topic selector

    Balances exploring new topics vs. practicing difficult ones.
    """

    @staticmethod
    def select(topics: List[Topic], total_questions: int, c: float = 1.4) -> Topic:
        """
        Select best topic using UCB1 algorithm.

        Args:
            topics: Available topics to choose from
            total_questions: Total questions answered so far
            c: Exploration parameter (higher = more exploration)

        Returns:
            Selected topic
        """
        if not topics:
            raise ValueError("No topics available for selection")

        # Always try untouched topics first
        for topic in topics:
            if topic.times_selected == 0:
                logger.debug(f"Selected unvisited topic: {topic.name}")
                return topic

        # UCB1 selection
        best_topic = None
        best_score = float('-inf')

        for topic in topics:
            # Exploitation: prefer topics not yet mastered
            exploitation = 1.0 - topic.bkt.p_mastery

            # Exploration: prefer topics not practiced recently
            exploration = c * math.sqrt(math.log(total_questions + 1) / topic.times_selected)

            score = exploitation + exploration

            if score > best_score:
                best_score = score
                best_topic = topic

        logger.debug(f"UCB selected: {best_topic.name} (score: {best_score:.3f})")
        return best_topic


# ==========================================================================================
# SUPABASE DATA ACCESS LAYER
# ==========================================================================================

class SupabaseDAO:
    """
    Data Access Object for Supabase operations.

    Handles all database interactions with error handling and logging.
    """

    def __init__(self, client: Client):
        self.client = client

    # ==================================================================================
    # USER OPERATIONS
    # ==================================================================================

    def create_user(self, username: str, email: Optional[str] = None) -> str:
        """
        Create new user in database.

        Args:
            username: Unique username
            email: Optional email address

        Returns:
            User ID (UUID as string)

        Raises:
            Exception: If user creation fails
        """
        try:
            result = self.client.table('users').insert({
                'username': username,
                'email': email,
                'created_at': datetime.now().isoformat()
            }).execute()

            user_id = result.data[0]['id']
            logger.info(f"Created user: {username} (ID: {user_id})")
            return user_id

        except Exception as e:
            logger.error(f"Failed to create user {username}: {e}")
            raise

    def get_user(self, user_id: str) -> Optional[Dict]:
        """Get user by ID"""
        try:
            result = self.client.table('users').select('*').eq('id', user_id).execute()
            return result.data[0] if result.data else None
        except Exception as e:
            logger.error(f"Failed to get user {user_id}: {e}")
            return None

    def get_user_by_username(self, username: str) -> Optional[Dict]:
        """Get user by username"""
        try:
            result = self.client.table('users').select('*').eq('username', username).execute()
            return result.data[0] if result.data else None
        except Exception as e:
            logger.error(f"Failed to get user by username {username}: {e}")
            return None

    def update_user_activity(self, user_id: str, total_questions: int):
        """Update user's last active time and question count"""
        try:
            self.client.table('users').update({
                'last_active': datetime.now().isoformat(),
                'total_questions': total_questions
            }).eq('id', user_id).execute()
        except Exception as e:
            logger.error(f"Failed to update user activity for {user_id}: {e}")

    # ==================================================================================
    # TOPIC OPERATIONS
    # ==================================================================================

    def create_topic(self, name: str, description: Optional[str] = None) -> str:
        """
        Create new topic.

        Args:
            name: Topic name
            description: Optional description

        Returns:
            Topic ID (UUID as string)
        """
        try:
            result = self.client.table('topics').insert({
                'name': name,
                'description': description,
                'created_at': datetime.now().isoformat()
            }).execute()

            topic_id = result.data[0]['id']
            logger.info(f"Created topic: {name} (ID: {topic_id})")
            return topic_id

        except Exception as e:
            logger.error(f"Failed to create topic {name}: {e}")
            raise

    def get_all_topics(self) -> List[Dict]:
        """Get all topics from database"""
        try:
            result = self.client.table('topics').select('*').execute()
            return result.data
        except Exception as e:
            logger.error(f"Failed to get all topics: {e}")
            return []

    def get_topic(self, topic_id: str) -> Optional[Dict]:
        """Get topic by ID"""
        try:
            result = self.client.table('topics').select('*').eq('id', topic_id).execute()
            return result.data[0] if result.data else None
        except Exception as e:
            logger.error(f"Failed to get topic {topic_id}: {e}")
            return None

    # ==================================================================================
    # USER_TOPIC OPERATIONS (BKT + FSRS State)
    # ==================================================================================

    def create_user_topic(self, user_id: str, topic_id: str) -> str:
        """
        Initialize user-topic relationship with default BKT/FSRS values.

        Args:
            user_id: User UUID
            topic_id: Topic UUID

        Returns:
            User-topic relationship ID
        """
        try:
            result = self.client.table('user_topics').insert({
                'user_id': user_id,
                'topic_id': topic_id,
                'created_at': datetime.now().isoformat()
            }).execute()

            ut_id = result.data[0]['id']
            logger.info(f"Created user-topic relationship: {ut_id}")
            return ut_id

        except Exception as e:
            logger.error(f"Failed to create user-topic for user {user_id}, topic {topic_id}: {e}")
            raise

    def get_user_topics(self, user_id: str) -> List[Dict]:
        """
        Get all topics for a user with their learning state.

        Args:
            user_id: User UUID

        Returns:
            List of user-topic records with topic details
        """
        try:
            result = self.client.table('user_topics').select(
                '*, topics(*)'
            ).eq('user_id', user_id).execute()

            return result.data

        except Exception as e:
            logger.error(f"Failed to get user topics for {user_id}: {e}")
            return []

    def get_user_topic(self, user_id: str, topic_id: str) -> Optional[Dict]:
        """Get specific user-topic state"""
        try:
            result = self.client.table('user_topics').select(
                '*'
            ).eq('user_id', user_id).eq('topic_id', topic_id).execute()

            return result.data[0] if result.data else None

        except Exception as e:
            logger.error(f"Failed to get user-topic for user {user_id}, topic {topic_id}: {e}")
            return None

    def update_user_topic_state(self, user_id: str, topic_id: str,
                                bkt: BKTParameters, fsrs: FSRSCard,
                                times_selected: int):
        """
        Update BKT and FSRS state for user-topic.

        Args:
            user_id: User UUID
            topic_id: Topic UUID
            bkt: Updated BKT parameters
            fsrs: Updated FSRS card state
            times_selected: Number of times topic has been selected
        """
        try:
            self.client.table('user_topics').update({
                # BKT state
                'bkt_p_mastery': bkt.p_mastery,
                'bkt_p_init': bkt.p_init,
                'bkt_p_learn': bkt.p_learn,
                'bkt_p_slip': bkt.p_slip,
                'bkt_p_guess': bkt.p_guess,

                # FSRS state
                'fsrs_stability': fsrs.stability,
                'fsrs_difficulty': fsrs.difficulty,
                'fsrs_reps': fsrs.reps,
                'fsrs_lapses': fsrs.lapses,
                'fsrs_last_review': fsrs.last_review,
                'fsrs_next_review_minutes': fsrs.next_review_minutes,

                # Metadata
                'times_selected': times_selected,
                'updated_at': datetime.now().isoformat()
            }).eq('user_id', user_id).eq('topic_id', topic_id).execute()

        except Exception as e:
            logger.error(f"Failed to update user-topic state: {e}")
            raise


    # ==================================================================================
    # REVIEW HISTORY OPERATIONS
    # ==================================================================================

    def add_review(self, user_id: str, topic_id: str, correct: bool,
                   quality_score: int, rating: Rating, mastery_before: float,
                   mastery_after: float, timestamp: datetime,
                   response_time_ms: Optional[int] = None):
        """
        Record a review in history.

        Args:
            user_id: User UUID
            topic_id: Topic UUID
            correct: Whether answer was correct
            quality_score: Quality score (0-5)
            rating: Performance rating
            mastery_before: Mastery probability before update
            mastery_after: Mastery probability after update
            timestamp: When review occurred
            response_time_ms: Response time in milliseconds
        """
        try:
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

        except Exception as e:
            logger.error(f"Failed to add review to history: {e}")
            # Don't raise - history is non-critical

    def get_user_review_history(self, user_id: str, limit: int = 100) -> List[Dict]:
        """Get recent review history for user"""
        try:
            result = self.client.table('review_history').select(
                '*, topics(name)'
            ).eq('user_id', user_id).order('timestamp', desc=True).limit(limit).execute()

            return result.data

        except Exception as e:
            logger.error(f"Failed to get user review history: {e}")
            return []

    def get_topic_review_history(self, user_id: str, topic_id: str) -> List[Dict]:
        """Get review history for specific topic"""
        try:
            result = self.client.table('review_history').select(
                '*'
            ).eq('user_id', user_id).eq('topic_id', topic_id).order('timestamp').execute()

            return result.data

        except Exception as e:
            logger.error(f"Failed to get topic review history: {e}")
            return []

    # ==================================================================================
    # ANALYTICS & STATS
    # ==================================================================================

    def get_user_stats(self, user_id: str) -> Dict:
        """
        Get comprehensive user statistics.

        Returns:
            Dictionary with user performance metrics
        """
        try:
            # Get user topics
            user_topics = self.get_user_topics(user_id)

            # Get recent reviews
            reviews = self.get_user_review_history(user_id, limit=1000)

            total_topics = len(user_topics)
            mastered_topics = sum(1 for ut in user_topics if ut['bkt_p_mastery'] >= 0.9)
            avg_mastery = sum(ut['bkt_p_mastery'] for ut in user_topics) / total_topics if total_topics > 0 else 0

            total_reviews = len(reviews)
            correct_reviews = sum(1 for r in reviews if r['correct'])
            accuracy = correct_reviews / total_reviews if total_reviews > 0 else 0

            # Calculate streak
            current_streak = 0
            for review in reviews:
                if review['correct']:
                    current_streak += 1
                else:
                    break

            return {
                'total_topics': total_topics,
                'mastered_topics': mastered_topics,
                'average_mastery': avg_mastery,
                'total_reviews': total_reviews,
                'accuracy': accuracy,
                'current_streak': current_streak
            }

        except Exception as e:
            logger.error(f"Failed to get user stats: {e}")
            return {}


# ==========================================================================================
# MAIN ADAPTIVE QUIZ SYSTEM
# ==========================================================================================

class AdaptiveQuizSystem:
    """
    Production-ready Adaptive Quiz System with Supabase backend.

    Features:
    - Bayesian Knowledge Tracing for mastery estimation
    - FSRS for optimal review scheduling
    - UCB for intelligent topic selection
    - Complete Supabase integration
    - Comprehensive logging and error handling

    Usage:
        # Initialize system
        client = SupabaseConfig.get_client()
        quiz = AdaptiveQuizSystem(client, user_id)

        # Start session
        quiz.start_session()

        # Get next topic
        topic = quiz.get_next_topic()

        # Submit answer
        result = quiz.submit_answer(topic.id, correct=True, quality_score=5)

        # Get summary
        summary = quiz.get_summary()
    """

    def __init__(self, supabase_client: Client, user_id: str):
        """
        Initialize adaptive quiz system for a user.

        Args:
            supabase_client: Authenticated Supabase client
            user_id: User UUID

        Raises:
            ValueError: If user not found
        """
        self.dao = SupabaseDAO(supabase_client)
        self.user_id = user_id
        self.topics: Dict[str, Topic] = {}
        self.total_questions = 0
        self.session_start: Optional[datetime] = None

        # Initialize engines
        self.bkt = BKTEngine()
        self.fsrs = FSRSEngine()
        self.ucb = UCBSelector()

        # Load user data
        self._load_user_data()

        logger.info(f"Initialized AdaptiveQuizSystem for user {user_id}")

    def _load_user_data(self):
        """Load user and their topic states from database"""
        user = self.dao.get_user(self.user_id)
        if not user:
            raise ValueError(f"User {self.user_id} not found in database")

        self.total_questions = user.get('total_questions', 0)

        # Load all user-topic relationships
        user_topics = self.dao.get_user_topics(self.user_id)

        for ut in user_topics:
            topic_data = ut['topics']

            self.topics[topic_data['id']] = Topic(
                id=topic_data['id'],
                name=topic_data['name'],
                description=topic_data.get('description'),
                created_at=topic_data.get('created_at'),
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

        logger.info(f"Loaded {len(self.topics)} topics for user {self.user_id}")

    def start_session(self):
        """Mark the start of a learning session"""
        self.session_start = datetime.now()
        logger.info("Learning session started")

    def get_next_topic(self, current_time: Optional[datetime] = None) -> Optional[Topic]:
        """
        Select next topic for the user using intelligent algorithms.

        Args:
            current_time: Current timestamp (defaults to now)

        Returns:
            Selected topic, or None if all topics are mastered
        """
        if current_time is None:
            current_time = datetime.now()

        # Filter to active topics (not mastered and due for review)
        active = []
        for topic in self.topics.values():
            if not self.bkt.is_mastered(topic.bkt):
                if self.fsrs.is_due(topic.fsrs, current_time):
                    active.append(topic)

        if not active:
            logger.info("No topics available (all mastered or not yet due)")
            return None

        # Use UCB to select best topic
        selected = self.ucb.select(active, self.total_questions)
        logger.info(f"Selected topic: {selected.name}")

        return selected

    def submit_answer(self, topic_id: str, correct: bool, quality_score: int,
                      timestamp: Optional[datetime] = None,
                      response_time_ms: Optional[int] = None) -> Dict:
        """
        Process student answer and update all learning state.

        Args:
            topic_id: Topic UUID
            correct: Whether answer was correct
            quality_score: Performance quality (0-5)
                0 = Complete blackout
                1 = Incorrect but familiar
                2 = Incorrect but close
                3 = Correct with difficulty
                4 = Correct after hesitation
                5 = Perfect recall
            timestamp: When answered (defaults to now)
            response_time_ms: Response time in milliseconds

        Returns:
            Result dictionary with updated metrics

        Raises:
            KeyError: If topic_id not found
        """
        if topic_id not in self.topics:
            raise KeyError(f"Topic {topic_id} not found for user {self.user_id}")

        topic = self.topics[topic_id]

        if timestamp is None:
            timestamp = datetime.now()

        # Convert quality score to rating
        rating = Rating.from_quality_score(quality_score)

        # Update BKT (mastery tracking)
        old_mastery = topic.bkt.p_mastery
        new_mastery = self.bkt.update(topic.bkt, correct)

        # Update FSRS (scheduling)
        topic.fsrs = self.fsrs.schedule(topic.fsrs, rating, timestamp)

        # Update selection count
        topic.times_selected += 1
        self.total_questions += 1

        # Persist to database
        try:
            self.dao.update_user_topic_state(
                self.user_id, topic_id, topic.bkt, topic.fsrs, topic.times_selected
            )

            self.dao.add_review(
                self.user_id, topic_id, correct, quality_score, rating,
                old_mastery, new_mastery, timestamp, response_time_ms
            )

            self.dao.update_user_activity(self.user_id, self.total_questions)

        except Exception as e:
            logger.error(f"Failed to persist answer: {e}")
            raise

        # Build result
        result = {
            'topic_id': topic_id,
            'topic_name': topic.name,
            'correct': correct,
            'quality_score': quality_score,
            'rating': rating.name,
            'mastery_before': old_mastery,
            'mastery_after': new_mastery,
            'mastery_change': new_mastery - old_mastery,
            'is_mastered': self.bkt.is_mastered(topic.bkt),
            'next_review_minutes': topic.fsrs.next_review_minutes,
            'next_review_hours': topic.fsrs.next_review_minutes / 60,
            'next_review_days': topic.fsrs.next_review_minutes / (60 * 24),
            'total_reviews': topic.fsrs.reps,
            'lapses': topic.fsrs.lapses,
            'difficulty': topic.fsrs.difficulty,
            'stability_days': topic.fsrs.stability
        }

        logger.info(f"Answer submitted: {topic.name} - {rating.name} - Mastery: {new_mastery:.1%}")

        return result

    def get_summary(self) -> Dict:
        """
        Get comprehensive session summary.

        Returns:
            Dictionary with session statistics
        """
        mastered = sum(1 for t in self.topics.values() if self.bkt.is_mastered(t.bkt))
        avg_mastery = sum(t.bkt.p_mastery for t in self.topics.values()) / len(self.topics) if self.topics else 0

        duration_min = 0.0
        if self.session_start:
            duration_min = (datetime.now() - self.session_start).total_seconds() / 60.0

        # Calculate topics by mastery level
        struggling = sum(1 for t in self.topics.values() if t.bkt.p_mastery < 0.3)
        learning = sum(1 for t in self.topics.values() if 0.3 <= t.bkt.p_mastery < 0.7)
        proficient = sum(1 for t in self.topics.values() if 0.7 <= t.bkt.p_mastery < 0.9)

        return {
            'total_questions': self.total_questions,
            'session_duration_minutes': duration_min,
            'topics_total': len(self.topics),
            'topics_mastered': mastered,
            'topics_proficient': proficient,
            'topics_learning': learning,
            'topics_struggling': struggling,
            'average_mastery': avg_mastery,
            'mastery_percentage': avg_mastery * 100
        }

    def get_topic_list(self, sort_by: str = 'mastery') -> List[Dict]:
        """
        Get list of all topics with their current state.

        Args:
            sort_by: How to sort ('mastery', 'name', 'reviews', 'due')

        Returns:
            List of topic dictionaries
        """
        topics_list = []

        for topic in self.topics.values():
            topics_list.append({
                'id': topic.id,
                'name': topic.name,
                'description': topic.description,
                'mastery': topic.bkt.p_mastery,
                'is_mastered': self.bkt.is_mastered(topic.bkt),
                'reviews': topic.fsrs.reps,
                'lapses': topic.fsrs.lapses,
                'difficulty': topic.fsrs.difficulty,
                'next_review_hours': topic.fsrs.next_review_minutes / 60,
                'is_due': self.fsrs.is_due(topic.fsrs, datetime.now()),
                'times_selected': topic.times_selected
            })

        # Sort
        if sort_by == 'mastery':
            topics_list.sort(key=lambda x: x['mastery'])
        elif sort_by == 'name':
            topics_list.sort(key=lambda x: x['name'])
        elif sort_by == 'reviews':
            topics_list.sort(key=lambda x: x['reviews'], reverse=True)
        elif sort_by == 'due':
            topics_list.sort(key=lambda x: (not x['is_due'], x['next_review_hours']))

        return topics_list

    def get_review_history(self, limit: int = 50) -> List[Dict]:
        """
        Get recent review history.

        Args:
            limit: Maximum number of reviews to return

        Returns:
            List of review records
        """
        return self.dao.get_user_review_history(self.user_id, limit)

    def get_stats(self) -> Dict:
        """
        Get comprehensive user statistics.

        Returns:
            Dictionary with user performance metrics
        """
        return self.dao.get_user_stats(self.user_id)


# ==========================================================================================
# USER MANAGEMENT FUNCTIONS
# ==========================================================================================

def create_user_with_topics(client: Client, username: str,
                            topic_names: List[str],
                            email: Optional[str] = None) -> str:
    """
    Create a new user and initialize their topics.

    Args:
        client: Supabase client
        username: Username for new user
        topic_names: List of topic names to initialize
        email: Optional email address

    Returns:
        User ID (UUID as string)
    """
    dao = SupabaseDAO(client)

    # Create user
    user_id = dao.create_user(username, email)
    logger.info(f"Created user: {username} (ID: {user_id})")

    # Get existing topics
    existing_topics = {t['name']: t['id'] for t in dao.get_all_topics()}

    # Create or link topics
    for topic_name in topic_names:
        if topic_name in existing_topics:
            topic_id = existing_topics[topic_name]
            logger.info(f"Using existing topic: {topic_name}")
        else:
            topic_id = dao.create_topic(topic_name)
            logger.info(f"Created new topic: {topic_name}")

        # Initialize user-topic relationship
        dao.create_user_topic(user_id, topic_id)

    logger.info(f"Initialized {len(topic_names)} topics for user {username}")
    return user_id


def get_or_create_user(client: Client, username: str,
                       topic_names: Optional[List[str]] = None,
                       email: Optional[str] = None) -> str:
    """
    Get existing user or create new one.

    Args:
        client: Supabase client
        username: Username to find or create
        topic_names: Topics to initialize for new users
        email: Email for new users

    Returns:
        User ID
    """
    dao = SupabaseDAO(client)

    # Try to find existing user
    user = dao.get_user_by_username(username)

    if user:
        logger.info(f"Found existing user: {username}")
        return user['id']
    else:
        logger.info(f"Creating new user: {username}")
        if topic_names:
            return create_user_with_topics(client, username, topic_names, email)
        else:
            return dao.create_user(username, email)


# ==========================================================================================
# CONVENIENCE FUNCTIONS
# ==========================================================================================

def initialize_quiz_system(
    username: str,
    topic_names: Optional[List[str]] = None,
    supabase_url: Optional[str] = None,
    supabase_key: Optional[str] = None
) -> AdaptiveQuizSystem:
    """
    Convenience function to initialize the complete quiz system.
    """
    # Set configuration directly if provided
    if supabase_url:
        SupabaseConfig.SUPABASE_URL = supabase_url
    if supabase_key:
        SupabaseConfig.SUPABASE_KEY = supabase_key

    # Get Supabase client
    client = SupabaseConfig.get_client()

    # Get or create user
    user_id = get_or_create_user(client, username, topic_names)

    # Initialize quiz system
    quiz = AdaptiveQuizSystem(client, user_id)

    logger.info("✅ Quiz system initialized successfully")
    return quiz

# ==========================================================================================
# EXAMPLE USAGE
# ==========================================================================================

def example_usage():
    """
    Example demonstrating production usage of the system.
    """
    print("=" * 80)
    print(" ADAPTIVE QUIZ SYSTEM - Production Example")
    print("=" * 80)
    print()

    # === CONFIGURATION ===
    SupabaseConfig.SUPABASE_URL = "url"
    SupabaseConfig.SUPABASE_KEY = "secret key"

    # Alternative: Use environment variables
    # os.environ['SUPABASE_URL'] = "https://..."
    # os.environ['SUPABASE_KEY'] = "sb_secret_..."

    try:
        # Initialize system
        quiz = initialize_quiz_system(
            username="student_001",
            topic_names=[
                "Variables & Data Types",
                "Control Flow (if/else)",
                "Loops (for/while)",
                "Functions & Parameters",
                "Lists & Arrays",
                "Dictionaries & Objects"
            ]
        )

        # ... rest of your example remains the same ...
        quiz.start_session()
        print(f"📚 Loaded {len(quiz.topics)} topics")

        # (Rest of the example code is unchanged)

    except Exception as e:
        logger.error(f"Error in example: {e}")
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    example_usage()