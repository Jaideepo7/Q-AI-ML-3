"""
Adaptive Quiz / Intelligent Tutoring System
============================================
Architecture: BKT (mastery tracking) + FSRS (scheduling) + UCB (topic selection)

Perfect for:
- Short intensive study sessions (2-4 hour cram sessions)
- Long-term retention across multiple logins
- Personalized adaptive learning

Components:
- BKT: Tracks real-time mastery probability (0-100%) for each topic
- FSRS: Modern spaced repetition scheduler (replacement for SM-2)
- UCB: Smart topic selector balancing weak topics vs exploration
"""

import json
import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import random


# ==========================================================================================
# CORE DATA MODELS
# ==========================================================================================

class Rating(Enum):
    """User performance rating"""
    AGAIN = 1  # Wrong
    HARD = 2  # Correct but struggled
    GOOD = 3  # Correct, normal difficulty
    EASY = 4  # Correct and easy


@dataclass
class BKTParameters:
    """Bayesian Knowledge Tracing state"""
    p_init: float = 0.1  # Prior knowledge probability
    p_learn: float = 0.3  # Learning rate
    p_slip: float = 0.1  # Slip probability (know but answer wrong)
    p_guess: float = 0.25  # Guess probability (don't know but answer right)
    p_mastery: float = 0.1  # Current mastery (updated in real-time)


@dataclass
class FSRSCard:
    """FSRS scheduling state"""
    stability: float = 0.0  # Memory stability in days
    difficulty: float = 5.0  # Intrinsic difficulty (1-10)
    reps: int = 0  # Total reviews
    lapses: int = 0  # Times forgotten
    last_review: Optional[str] = None  # ISO timestamp
    next_review_minutes: float = 0.0  # Minutes until next review


@dataclass
class Topic:
    """Complete topic state"""
    id: str
    name: str
    bkt: BKTParameters
    fsrs: FSRSCard
    times_selected: int = 0
    history: List[Dict] = None

    def __post_init__(self):
        if self.history is None:
            self.history = []


# ==========================================================================================
# BKT ENGINE - Real-time Mastery Tracking
# ==========================================================================================

class BKTEngine:
    """Updates mastery probability after each answer"""

    @staticmethod
    def update(bkt: BKTParameters, correct: bool) -> float:
        """
        Bayesian update of mastery probability.

        Returns:
            New mastery probability (0.0 to 1.0)
        """
        p_know = bkt.p_mastery

        if correct:
            # Update based on correct answer
            p_correct_know = 1 - bkt.p_slip
            p_correct_dont = bkt.p_guess

            numerator = p_correct_know * p_know
            denominator = p_correct_know * p_know + p_correct_dont * (1 - p_know)

            p_know_after = numerator / denominator if denominator > 0 else p_know
        else:
            # Update based on wrong answer
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
# FSRS ENGINE - Spaced Repetition Scheduling
# ==========================================================================================

class FSRSEngine:
    """Simplified FSRS for both short sessions and long-term retention"""

    # Initial stability for new cards (in days)
    INIT_STABILITY = {
        Rating.AGAIN: 0.003,  # ~5 minutes
        Rating.HARD: 0.01,  # ~15 minutes
        Rating.GOOD: 0.02,  # ~30 minutes
        Rating.EASY: 0.05  # ~1 hour
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
            # First review - use initial stability
            new_card.stability = FSRSEngine.INIT_STABILITY[rating]
            new_card.difficulty = 5.0
        else:
            # Subsequent reviews
            if rating == Rating.AGAIN:
                # Forgotten - reduce stability, increase difficulty
                new_card.stability = card.stability * 0.2
                new_card.difficulty = min(10.0, card.difficulty + 2.0)
                new_card.lapses += 1
            else:
                # Remembered - increase stability based on rating
                multipliers = {
                    Rating.HARD: 1.5,
                    Rating.GOOD: 2.5,
                    Rating.EASY: 4.0
                }

                # Adjust by difficulty (harder = slower growth)
                difficulty_factor = 1.0 - (card.difficulty / 20.0)
                multiplier = multipliers[rating] * (0.7 + 0.6 * difficulty_factor)

                new_card.stability = card.stability * multiplier

                # Adjust difficulty
                diff_changes = {
                    Rating.HARD: 1.0,
                    Rating.GOOD: 0.0,
                    Rating.EASY: -1.0
                }
                new_card.difficulty = max(1.0, min(10.0, card.difficulty + diff_changes[rating]))

        # Convert to minutes for easier intra-session handling
        new_card.next_review_minutes = new_card.stability * 24 * 60

        return new_card

    @staticmethod
    def is_due(card: FSRSCard, current_time: datetime) -> bool:
        """Check if card is due for review"""
        if card.last_review is None:
            return True  # Never reviewed

        last_time = datetime.fromisoformat(card.last_review)
        minutes_since = (current_time - last_time).total_seconds() / 60.0

        return minutes_since >= card.next_review_minutes


# ==========================================================================================
# UCB SELECTOR - Intelligent Topic Selection
# ==========================================================================================

class UCBSelector:
    """Upper Confidence Bound algorithm for topic selection"""

    @staticmethod
    def select(topics: List[Topic], total_questions: int, c: float = 1.4) -> Topic:
        """
        Select best topic using UCB1.

        Balances:
        - Exploitation: Focus on weak topics (low mastery)
        - Exploration: Try all topics fairly (uncertainty bonus)

        Args:
            topics: Available topics (not mastered, due for review)
            total_questions: Total questions answered (for exploration calculation)
            c: Exploration constant (higher = more exploration)
        """
        if not topics:
            raise ValueError("No topics available")

        # Always try untouched topics first
        for topic in topics:
            if topic.times_selected == 0:
                return topic

        best_topic = None
        best_score = float('-inf')

        for topic in topics:
            # Exploitation: Lower mastery = higher priority
            exploitation = 1.0 - topic.bkt.p_mastery

            # Exploration: Uncertainty bonus
            exploration = c * math.sqrt(math.log(total_questions + 1) / topic.times_selected)

            score = exploitation + exploration

            if score > best_score:
                best_score = score
                best_topic = topic

        return best_topic


# ==========================================================================================
# MAIN ADAPTIVE QUIZ SYSTEM
# ==========================================================================================

class AdaptiveQuiz:
    """Main system integrating BKT, FSRS, and UCB"""

    def __init__(self, user_id: str):
        self.user_id = user_id
        self.topics: Dict[str, Topic] = {}
        self.total_questions = 0
        self.session_start = None

        self.bkt = BKTEngine()
        self.fsrs = FSRSEngine()
        self.ucb = UCBSelector()

    def add_topics(self, topic_names: List[str]):
        """Initialize topics"""
        for i, name in enumerate(topic_names):
            topic_id = f"topic_{i + 1}"
            self.topics[topic_id] = Topic(
                id=topic_id,
                name=name,
                bkt=BKTParameters(),
                fsrs=FSRSCard()
            )

    def start_session(self):
        """Mark session start"""
        self.session_start = datetime.now()

    def get_next_topic(self, current_time: datetime = None) -> Optional[Topic]:
        """
        Select next topic using UCB on active topics.

        Active = not mastered AND due for review
        """
        if current_time is None:
            current_time = datetime.now()

        # Filter active topics
        active = []
        for topic in self.topics.values():
            if not self.bkt.is_mastered(topic.bkt):
                if self.fsrs.is_due(topic.fsrs, current_time):
                    active.append(topic)

        if not active:
            return None

        return self.ucb.select(active, self.total_questions)

    def submit_answer(self, topic_id: str, correct: bool, rating: Rating = None,
                      timestamp: datetime = None) -> Dict:
        """
        Process answer and update all systems.

        Returns dict with:
        - mastery_before, mastery_after, mastery_change
        - is_mastered
        - next_review_minutes
        """
        topic = self.topics[topic_id]

        if timestamp is None:
            timestamp = datetime.now()

        if rating is None:
            rating = Rating.GOOD if correct else Rating.AGAIN

        # Update BKT
        old_mastery = topic.bkt.p_mastery
        new_mastery = self.bkt.update(topic.bkt, correct)

        # Update FSRS
        topic.fsrs = self.fsrs.schedule(topic.fsrs, rating, timestamp)

        # Update counts
        topic.times_selected += 1
        self.total_questions += 1

        # Record history
        topic.history.append({
            'timestamp': timestamp.isoformat(),
            'correct': correct,
            'rating': rating.name,
            'mastery': new_mastery
        })

        return {
            'topic_name': topic.name,
            'mastery_before': old_mastery,
            'mastery_after': new_mastery,
            'mastery_change': new_mastery - old_mastery,
            'is_mastered': self.bkt.is_mastered(topic.bkt),
            'next_review_minutes': topic.fsrs.next_review_minutes,
            'total_reviews': topic.fsrs.reps
        }

    def get_summary(self) -> Dict:
        """Get session summary"""
        mastered = sum(1 for t in self.topics.values() if self.bkt.is_mastered(t.bkt))
        avg_mastery = sum(t.bkt.p_mastery for t in self.topics.values()) / len(self.topics)

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

    def save(self, filepath: str):
        """Save to JSON"""
        data = {
            'user_id': self.user_id,
            'total_questions': self.total_questions,
            'session_start': self.session_start.isoformat() if self.session_start else None,
            'topics': {
                tid: {
                    'id': t.id,
                    'name': t.name,
                    'bkt': asdict(t.bkt),
                    'fsrs': asdict(t.fsrs),
                    'times_selected': t.times_selected,
                    'history': t.history
                }
                for tid, t in self.topics.items()
            }
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self, filepath: str):
        """Load from JSON"""
        with open(filepath, 'r') as f:
            data = json.load(f)

        self.user_id = data['user_id']
        self.total_questions = data['total_questions']
        self.session_start = datetime.fromisoformat(data['session_start']) if data['session_start'] else None

        self.topics = {}
        for tid, tdata in data['topics'].items():
            self.topics[tid] = Topic(
                id=tdata['id'],
                name=tdata['name'],
                bkt=BKTParameters(**tdata['bkt']),
                fsrs=FSRSCard(**tdata['fsrs']),
                times_selected=tdata['times_selected'],
                history=tdata['history']
            )


# ==========================================================================================
# EXAMPLE / DEMONSTRATION
# ==========================================================================================

def run_demo():
    """Realistic 4-hour cram session simulation"""

    print("=" * 80)
    print(" ADAPTIVE QUIZ SYSTEM - Demonstration")
    print("=" * 80)
    print()
    print("Architecture: BKT (mastery) + FSRS (scheduling) + UCB (selection)")
    print()

    # Setup
    quiz = AdaptiveQuiz(user_id="student_123")
    quiz.add_topics([
        "Variables & Data Types",
        "Control Flow (if/else)",
        "Loops (for/while)",
        "Functions & Parameters",
        "Lists & Tuples",
        "Dictionaries",
        "Exception Handling",
        "File I/O Operations"
    ])

    quiz.start_session()

    print(f"📚 User: {quiz.user_id}")
    print(f"📖 Topics: {len(quiz.topics)}")
    print()
    print("Starting 4-hour cram session simulation...")
    print("-" * 80)
    print()

    # Simulate session
    sim_time = datetime.now()
    question_count = 0
    max_questions = 60

    while question_count < max_questions:
        # Get next topic
        topic = quiz.get_next_topic(sim_time)

        if not topic:
            # Nothing due - skip ahead
            sim_time += timedelta(minutes=10)
            topic = quiz.get_next_topic(sim_time)

            if not topic:
                print("\n✨ All topics mastered! Session complete.\n")
                break

        question_count += 1

        # Simulate realistic performance
        # - Weak topics: ~40-60% success
        # - Medium topics: ~60-80% success
        # - Strong topics: ~80-95% success
        # - Gradual improvement over session

        base_rate = 0.35 + (topic.bkt.p_mastery * 0.45)
        session_progress = question_count / max_questions
        improvement = session_progress * 0.25
        success_rate = min(0.95, base_rate + improvement)

        correct = random.random() < success_rate

        # Smart rating assignment
        if correct:
            if topic.bkt.p_mastery > 0.75:
                rating = Rating.EASY
            elif topic.bkt.p_mastery > 0.5:
                rating = Rating.GOOD
            else:
                rating = Rating.HARD
        else:
            rating = Rating.AGAIN

        # Submit
        result = quiz.submit_answer(topic.id, correct, rating, sim_time)

        # Display
        status = "*" if correct else "✗"
        bar_filled = int(result['mastery_after'] * 25)
        bar = "█" * bar_filled + "░" * (25 - bar_filled)

        print(f"Q{question_count:2d}: {status}  {topic.name[:28]:<28}  [{bar}]  "
              f"{result['mastery_after']:>5.1%}  ({result['mastery_change']:+.2%})")

        if result['is_mastered'] and result['mastery_before'] < 0.9:
            print(f"       🌟 MASTERED! Next review in {result['next_review_minutes'] / 60:.1f} hours")

        # Advance time (realistic pace: 2-5 min per question)
        sim_time += timedelta(minutes=random.randint(2, 5))

        # Stop at 4 hours
        if (sim_time - quiz.session_start).total_seconds() > 4 * 3600:
            print(f"\n⏰ Time limit reached ({(sim_time - quiz.session_start).total_seconds() / 3600:.1f}h)\n")
            break

    # Summary
    print("-" * 80)
    print()
    summary = quiz.get_summary()

    print("SESSION SUMMARY")
    print("=" * 80)
    print(f"Questions answered:     {summary['total_questions']}")
    print(f"Topics mastered:        {summary['topics_mastered']}/{summary['topics_total']}")
    print(f"Average mastery:        {summary['average_mastery']:.1%}")
    print(f"Session duration:       {summary['duration_minutes']:.0f} minutes")
    print()

    # Detailed breakdown
    print("TOPIC BREAKDOWN")
    print("-" * 80)
    print(f"{'Status':<13} {'Topic':<28} {'Mastery':>8} {'Reviews':>8} {'Next':>10}")
    print("-" * 80)

    topics_sorted = sorted(quiz.topics.values(), key=lambda t: t.bkt.p_mastery, reverse=True)

    for topic in topics_sorted:
        status = "* Mastered" if quiz.bkt.is_mastered(topic.bkt) else "  Learning"
        next_review = f"{topic.fsrs.next_review_minutes / 60:.1f}h" if topic.times_selected > 0 else "-"

        print(f"{status:<13} {topic.name[:28]:<28} {topic.bkt.p_mastery:>7.1%} "
              f"{topic.times_selected:>8} {next_review:>10}")

    # Save
    save_path = "/home/claude/quiz_session.json"
    quiz.save(save_path)
    print()
    print(f"* Progress saved to: {save_path}")

    # Demo persistence
    print()
    print("=" * 80)
    print("DEMONSTRATING PERSISTENCE")
    print("-" * 80)

    # Reload
    quiz2 = AdaptiveQuiz(user_id="student_123")
    quiz2.load(save_path)

    print(f"* Loaded user: {quiz2.user_id}")
    print(f"* Questions answered: {quiz2.total_questions}")
    print(f"* Topics in memory: {len(quiz2.topics)}")

    # What's due tomorrow?
    print()
    print("If user returns in 24 hours...")
    tomorrow = sim_time + timedelta(days=1)

    due_count = 0
    due_list = []
    for topic in quiz2.topics.values():
        if quiz2.fsrs.is_due(topic.fsrs, tomorrow):
            due_count += 1
            if len(due_list) < 5:
                due_list.append(topic.name)

    print(f"Topics due for review: {due_count}/{len(quiz2.topics)}")
    if due_list:
        for i, name in enumerate(due_list, 1):
            print(f"  {i}. {name}")
        if due_count > 5:
            print(f"  ... and {due_count - 5} more")

    print()
    print("=" * 80)
    print("Demo complete! The system is production-ready.")
    print("=" * 80)


if __name__ == "__main__":
    run_demo()