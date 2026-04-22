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
Version: 2.1 - Environment Variables Support
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

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()  # This loads variables from .env file into os.environ
except ImportError:
    logging.warning("python-dotenv not installed. Install with: pip install python-dotenv")
    logging.warning("You can still use environment variables without it.")

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
    Supabase Configuration with Environment Variable Support

    Credentials are loaded from:
    1. .env file (recommended for development)
    2. Environment variables (recommended for production)
    3. Direct assignment (not recommended - deprecated)
    """
    # These will be None - we'll read from environment instead
    SUPABASE_URL: Optional[str] = None
    SUPABASE_KEY: Optional[str] = None

    @classmethod
    def get_url(cls) -> str:
        """
        Get Supabase URL from environment variables or .env file.

        Priority:
        1. Environment variable SUPABASE_URL
        2. Class variable (deprecated)

        Returns:
            Supabase URL

        Raises:
            ValueError: If URL is not set
        """
        url = os.getenv('SUPABASE_URL') or cls.SUPABASE_URL
        if not url:
            raise ValueError(
                "SUPABASE_URL is not set!\n"
                "Set it in one of these ways:\n"
                "1. Create a .env file with: SUPABASE_URL=your-url\n"
                "2. Export environment variable: export SUPABASE_URL='your-url'\n"
                "3. Set in code (not recommended): SupabaseConfig.SUPABASE_URL = 'your-url'"
            )
        return url.rstrip('/')

    @classmethod
    def get_key(cls) -> str:
        """
        Get Supabase API key from environment variables or .env file.

        Priority:
        1. Environment variable SUPABASE_KEY
        2. Class variable (deprecated)

        Returns:
            Supabase API key

        Raises:
            ValueError: If key is not set
        """
        key = os.getenv('SUPABASE_KEY') or cls.SUPABASE_KEY
        if not key:
            raise ValueError(
                "SUPABASE_KEY is not set!\n"
                "Set it in one of these ways:\n"
                "1. Create a .env file with: SUPABASE_KEY=your-key\n"
                "2. Export environment variable: export SUPABASE_KEY='your-key'\n"
                "3. Set in code (not recommended): SupabaseConfig.SUPABASE_KEY = 'your-key'"
            )
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

    @classmethod
    def check_configuration(cls) -> bool:
        """
        Check if configuration is properly set up.

        Returns:
            True if configuration is valid, False otherwise
        """
        try:
            url = cls.get_url()
            key = cls.get_key()
            logger.info("✅ Configuration check passed")
            logger.info(f"   Supabase URL: {url}")
            logger.info(f"   API Key: {key[:20]}...{key[-4:]}")
            return True
        except ValueError as e:
            logger.error(f"❌ Configuration check failed: {e}")
            return False


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
            card: Current FSRS card state
            rating: User's performance rating
            timestamp: Current timestamp

        Returns:
            Updated FSRS card
        """
        if card.reps == 0:
            # First review - use initial stability
            card.stability = FSRSEngine.INIT_STABILITY[rating]
            card.difficulty = 5.0  # Start at medium difficulty
        else:
            # Subsequent reviews - update stability and difficulty
            card.stability = FSRSEngine._update_stability(card, rating)
            card.difficulty = FSRSEngine._update_difficulty(card, rating)

        # Handle lapses
        if rating == Rating.AGAIN:
            card.lapses += 1

        # Calculate next review interval
        card.next_review_minutes = FSRSEngine._calculate_interval(card.stability)
        card.reps += 1
        card.last_review = timestamp.isoformat()

        return card

    @staticmethod
    def _update_stability(card: FSRSCard, rating: Rating) -> float:
        """Update memory stability based on rating"""
        retrievability = 0.9  # Assume 90% retrievability at review time

        if rating == Rating.AGAIN:
            # Failed recall - stability decreases
            new_stability = card.stability * 0.3
        elif rating == Rating.HARD:
            # Difficult recall - modest increase
            new_stability = card.stability * (1.2 + 0.1 * retrievability)
        elif rating == Rating.GOOD:
            # Normal recall - good increase
            new_stability = card.stability * (2.5 + 0.5 * retrievability)
        else:  # EASY
            # Easy recall - large increase
            new_stability = card.stability * (3.5 + 1.0 * retrievability)

        # Apply difficulty factor
        difficulty_factor = 1.0 - (card.difficulty - 5.0) * 0.05
        new_stability *= difficulty_factor

        return max(new_stability, 0.001)  # Minimum stability

    @staticmethod
    def _update_difficulty(card: FSRSCard, rating: Rating) -> float:
        """Update item difficulty based on rating"""
        difficulty_change = {
            Rating.AGAIN: 0.8,   # Increase difficulty
            Rating.HARD: 0.3,
            Rating.GOOD: -0.1,
            Rating.EASY: -0.3    # Decrease difficulty
        }

        new_difficulty = card.difficulty + difficulty_change[rating]
        return max(1.0, min(10.0, new_difficulty))  # Clamp to [1, 10]

    @staticmethod
    def _calculate_interval(stability: float) -> float:
        """
        Calculate review interval in minutes from stability.

        Args:
            stability: Memory stability in days

        Returns:
            Interval in minutes
        """
        # Convert days to minutes
        return stability * 24 * 60

    @staticmethod
    def is_due(card: FSRSCard, current_time: datetime) -> bool:
        """
        Check if a card is due for review.

        Args:
            card: FSRS card
            current_time: Current datetime

        Returns:
            True if card is due for review
        """
        if card.last_review is None:
            return True

        last_review = datetime.fromisoformat(card.last_review)
        next_review = last_review + timedelta(minutes=card.next_review_minutes)

        return current_time >= next_review


# ==========================================================================================
# DATABASE ACCESS OBJECT (DAO)
# ==========================================================================================

class SupabaseDAO:
    """Data Access Object for Supabase operations"""

    def __init__(self, client: Client):
        self.client = client

    # User operations
    def create_user(self, username: str, email: Optional[str] = None) -> str:
        """Create a new user"""
        data = {'username': username}
        if email:
            data['email'] = email

        result = self.client.table('users').insert(data).execute()
        return result.data[0]['id']

    def get_user_by_username(self, username: str) -> Optional[Dict]:
        """Get user by username"""
        result = self.client.table('users').select('*').eq('username', username).execute()
        return result.data[0] if result.data else None

    # Topic operations
    def get_all_topics(self) -> List[Dict]:
        """Get all available topics"""
        result = self.client.table('topics').select('*').execute()
        return result.data

    def create_topic(self, name: str, description: Optional[str] = None) -> str:
        """Create a new topic"""
        data = {'name': name}
        if description:
            data['description'] = description

        result = self.client.table('topics').insert(data).execute()
        return result.data[0]['id']

    # User-Topic operations
    def get_user_topics(self, user_id: str) -> List[Dict]:
        """Get all topics for a user with their learning state"""
        result = (
            self.client.table('user_topics')
            .select('*, topics(*)')
            .eq('user_id', user_id)
            .execute()
        )
        return result.data

    def create_user_topic(self, user_id: str, topic_id: str) -> str:
        """Initialize a topic for a user"""
        data = {
            'user_id': user_id,
            'topic_id': topic_id
        }
        result = self.client.table('user_topics').insert(data).execute()
        return result.data[0]['id']

    def update_user_topic(self, user_id: str, topic_id: str,
                         bkt_state: Dict, fsrs_state: Dict,
                         times_selected: int) -> None:
        """Update user's topic state"""
        data = {
            'bkt_state': bkt_state,
            'fsrs_state': fsrs_state,
            'times_selected': times_selected,
            'updated_at': datetime.now().isoformat()
        }

        self.client.table('user_topics').update(data).match({
            'user_id': user_id,
            'topic_id': topic_id
        }).execute()

    # Review history operations
    def record_review(self, user_id: str, topic_id: str,
                     correct: bool, quality_score: int, rating: str,
                     mastery_before: float, mastery_after: float,
                     difficulty_before: float, difficulty_after: float) -> None:
        """Record a review in history"""
        data = {
            'user_id': user_id,
            'topic_id': topic_id,
            'correct': correct,
            'quality_score': quality_score,
            'rating': rating,
            'mastery_before': mastery_before,
            'mastery_after': mastery_after,
            'difficulty_before': difficulty_before,
            'difficulty_after': difficulty_after
        }

        self.client.table('review_history').insert(data).execute()

    def get_user_review_history(self, user_id: str, limit: int = 50) -> List[Dict]:
        """Get user's review history"""
        result = (
            self.client.table('review_history')
            .select('*, topics(name)')
            .eq('user_id', user_id)
            .order('review_timestamp', desc=True)
            .limit(limit)
            .execute()
        )
        return result.data

    def get_user_stats(self, user_id: str) -> Dict:
        """Get comprehensive user statistics"""
        # Get all reviews
        reviews = self.client.table('review_history').select('*').eq('user_id', user_id).execute()

        if not reviews.data:
            return {
                'total_reviews': 0,
                'correct_count': 0,
                'accuracy': 0.0,
                'avg_quality': 0.0
            }

        total = len(reviews.data)
        correct = sum(1 for r in reviews.data if r['correct'])
        avg_quality = sum(r['quality_score'] for r in reviews.data) / total

        return {
            'total_reviews': total,
            'correct_count': correct,
            'accuracy': correct / total if total > 0 else 0,
            'avg_quality': avg_quality
        }


# ==========================================================================================
# UCB TOPIC SELECTOR
# ==========================================================================================

class UCBTopicSelector:
    """
    Upper Confidence Bound (UCB) algorithm for topic selection.

    Balances exploitation (studying weak topics) vs exploration (trying all topics).
    """

    @staticmethod
    def select_topic(topics: Dict[str, Topic], total_selections: int,
                    fsrs_engine: FSRSEngine, current_time: datetime,
                    exploration_constant: float = 1.0) -> Optional[str]:
        """
        Select next topic using UCB algorithm with FSRS due checking.

        Args:
            topics: Dictionary of available topics
            total_selections: Total number of topic selections made
            fsrs_engine: FSRS engine for due checking
            current_time: Current datetime
            exploration_constant: Balance between exploitation and exploration

        Returns:
            Topic ID to study, or None if no topics available
        """
        if not topics:
            return None

        # Filter to only due topics
        due_topics = {
            tid: topic for tid, topic in topics.items()
            if fsrs_engine.is_due(topic.fsrs, current_time)
        }

        if not due_topics:
            return None

        best_topic = None
        best_score = float('-inf')

        for topic_id, topic in due_topics.items():
            # Exploitation: Lower mastery = higher priority
            exploitation = 1.0 - topic.bkt.p_mastery

            # Exploration: Less selected topics get bonus
            if topic.times_selected == 0:
                exploration = float('inf')  # Always try unexplored topics
            else:
                exploration = math.sqrt(
                    (2 * math.log(total_selections + 1)) / topic.times_selected
                )

            # UCB score
            ucb_score = exploitation + exploration_constant * exploration

            if ucb_score > best_score:
                best_score = ucb_score
                best_topic = topic_id

        return best_topic


# ==========================================================================================
# MAIN ADAPTIVE QUIZ SYSTEM
# ==========================================================================================

class AdaptiveQuizSystem:
    """
    Main adaptive quiz system coordinating all components.
    """

    def __init__(self, client: Client, user_id: str):
        """
        Initialize the adaptive quiz system.

        Args:
            client: Supabase client
            user_id: User ID
        """
        self.client = client
        self.user_id = user_id
        self.dao = SupabaseDAO(client)

        # Engines
        self.bkt = BKTEngine()
        self.fsrs = FSRSEngine()
        self.ucb = UCBTopicSelector()

        # State
        self.topics: Dict[str, Topic] = {}
        self.total_questions = 0
        self.session_start: Optional[datetime] = None

        # Load user topics
        self._load_topics()

    def _load_topics(self):
        """Load user's topics from database"""
        user_topics = self.dao.get_user_topics(self.user_id)

        for ut in user_topics:
            topic_data = ut['topics']

            # Parse BKT state
            bkt_state = ut['bkt_state']
            bkt = BKTParameters(**bkt_state)

            # Parse FSRS state
            fsrs_state = ut['fsrs_state']
            fsrs = FSRSCard(**fsrs_state)

            # Create Topic object
            topic = Topic(
                id=topic_data['id'],
                name=topic_data['name'],
                description=topic_data.get('description'),
                bkt=bkt,
                fsrs=fsrs,
                times_selected=ut['times_selected'],
                created_at=ut['created_at']
            )

            self.topics[topic.id] = topic

        logger.info(f"Loaded {len(self.topics)} topics for user {self.user_id}")

    def start_session(self):
        """Start a new learning session"""
        self.session_start = datetime.now()
        logger.info(f"Started session at {self.session_start}")

    def select_next_topic(self) -> Optional[str]:
        """
        Select the next topic to study.

        Returns:
            Topic ID or None if no topics due
        """
        topic_id = self.ucb.select_topic(
            self.topics,
            self.total_questions,
            self.fsrs,
            datetime.now()
        )

        if topic_id:
            self.topics[topic_id].times_selected += 1
            self.total_questions += 1

        return topic_id

    def submit_answer(self, topic_id: str, correct: bool, quality_score: int) -> Dict:
        """
        Submit an answer and update all states.

        Args:
            topic_id: ID of the topic answered
            correct: Whether the answer was correct
            quality_score: Quality of recall (0-5)

        Returns:
            Dictionary with update results
        """
        topic = self.topics[topic_id]
        rating = Rating.from_quality_score(quality_score)

        # Capture before states
        mastery_before = topic.bkt.p_mastery
        difficulty_before = topic.fsrs.difficulty

        # Update BKT
        new_mastery = self.bkt.update(topic.bkt, correct)

        # Update FSRS
        topic.fsrs = self.fsrs.schedule(topic.fsrs, rating, datetime.now())

        # Capture after states
        mastery_after = topic.bkt.p_mastery
        difficulty_after = topic.fsrs.difficulty

        # Save to database
        self.dao.update_user_topic(
            self.user_id,
            topic_id,
            topic.bkt.to_dict(),
            topic.fsrs.to_dict(),
            topic.times_selected
        )

        # Record in history
        self.dao.record_review(
            self.user_id,
            topic_id,
            correct,
            quality_score,
            rating.name,
            mastery_before,
            mastery_after,
            difficulty_before,
            difficulty_after
        )

        logger.info(
            f"Topic {topic.name}: "
            f"Mastery {mastery_before:.2%} → {mastery_after:.2%}, "
            f"Next review: {topic.fsrs.next_review_minutes:.0f} min"
        )

        return {
            'topic_id': topic_id,
            'topic_name': topic.name,
            'correct': correct,
            'quality_score': quality_score,
            'rating': rating.name,
            'old_mastery': mastery_before,
            'new_mastery': mastery_after,
            'old_difficulty': difficulty_before,
            'new_difficulty': difficulty_after,
            'next_review_minutes': topic.fsrs.next_review_minutes
        }

    def get_session_summary(self) -> Dict:
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

    Args:
        username: Username for the student
        topic_names: List of topics to initialize (optional)
        supabase_url: Supabase URL (optional, reads from .env if not provided)
        supabase_key: Supabase API key (optional, reads from .env if not provided)

    Returns:
        Initialized AdaptiveQuizSystem

    Note:
        If supabase_url or supabase_key are provided, they will override
        environment variables. However, it's recommended to use .env file instead.
    """
    # Set configuration directly if provided (temporary override)
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
    # Method 1: Using .env file (RECOMMENDED)
    # Create a .env file with:
    # SUPABASE_URL=https://your-project.supabase.co
    # SUPABASE_KEY=your-anon-key

    # Method 2: Direct assignment (NOT RECOMMENDED - deprecated)
    # SupabaseConfig.SUPABASE_URL = "your-url"
    # SupabaseConfig.SUPABASE_KEY = "your-key"

    # Method 3: Environment variables
    # export SUPABASE_URL="your-url"
    # export SUPABASE_KEY="your-key"

    # Check configuration
    if not SupabaseConfig.check_configuration():
        print("\n❌ Please configure your environment variables!")
        print("\nCreate a .env file with:")
        print("SUPABASE_URL=https://your-project.supabase.co")
        print("SUPABASE_KEY=your-anon-key")
        return

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

        # Start session
        quiz.start_session()
        print(f"📚 Loaded {len(quiz.topics)} topics")
        print()

        # Example quiz loop
        for i in range(5):
            # Select topic
            topic_id = quiz.select_next_topic()
            if not topic_id:
                print("No topics due for review!")
                break

            topic = quiz.topics[topic_id]
            print(f"\n{'='*70}")
            print(f"Question {i+1}: {topic.name}")
            print(f"Current mastery: {topic.bkt.p_mastery:.1%}")
            print(f"{'='*70}")

            # Simulate question and answer
            # In real usage, you would:
            # 1. Generate a question about the topic
            # 2. Present it to the user
            # 3. Get their answer and confidence

            # For demo, simulate a correct answer with quality 4
            result = quiz.submit_answer(topic_id, correct=True, quality_score=4)

            print(f"\n✅ Answer recorded!")
            print(f"   Mastery: {result['old_mastery']:.1%} → {result['new_mastery']:.1%}")
            print(f"   Next review: {result['next_review_minutes']:.0f} minutes")

        # Show summary
        print("\n" + "="*70)
        print(" SESSION SUMMARY")
        print("="*70)
        summary = quiz.get_session_summary()
        print(f"\n📊 Questions answered: {summary['total_questions']}")
        print(f"⏱️  Duration: {summary['session_duration_minutes']:.1f} minutes")
        print(f"🎯 Topics mastered: {summary['topics_mastered']}/{summary['topics_total']}")
        print(f"📈 Average mastery: {summary['average_mastery']:.1%}")
        print()

    except Exception as e:
        logger.error(f"Error in example: {e}")
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    example_usage()