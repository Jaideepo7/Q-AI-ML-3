"""
ADAPTIVE QUIZ SYSTEM - WITH GEMINI QUESTION GENERATION
======================================================
Uses your quiz_generator.py (Gemini + spaCy) for questions
Integrates with FSRS scheduling from Quiz_Two_Enhanced.py

Dependencies:
    pip install google-genai spacy python-dotenv supabase
    python -m spacy download en_core_web_sm

Usage:
    python run_quiz_gemini.py
"""

import os
import sys
import json
from datetime import datetime
from typing import Dict, List, Optional

# Load environment variables from .env file
try:
    from dotenv import load_dotenv

    load_dotenv()
    print("✅ Loaded environment variables from .env file")
except ImportError:
    print("⚠️  python-dotenv not installed. Install with: pip install python-dotenv")
    print("   You can still use environment variables without it.\n")

# Import your quiz generator
try:
    from quiz_generator import generate_quiz
except ImportError:
    print("❌ quiz_generator.py not found!")
    print("   Make sure quiz_generator.py is in the same directory")
    sys.exit(1)

# Import from Quiz_Two_Enhanced.py
from Quiz_Engine import (
    initialize_quiz_system,
    SupabaseConfig,
    Rating,
    AdaptiveQuizSystem
)


class GeminiQuestionGenerator:
    """
    Generates questions using Gemini API via quiz_generator.py
    Adapts to FSRS difficulty and topic mastery
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Gemini question generator.

        Args:
            api_key: Gemini API key (reads from .env if not provided)
        """
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY is not set!\n"
                "Add it to your .env file:\n"
                "GEMINI_API_KEY=your-api-key-here"
            )

        # Cache to store generated quizzes per topic
        self.quiz_cache: Dict[str, List[Dict]] = {}
        self.current_question_index: Dict[str, int] = {}

    def generate_topic_context(
            self,
            topic_name: str,
            difficulty: float,
            mastery_level: float
    ) -> str:
        """
        Generate a contextual "transcript" based on the topic.
        This simulates having transcript content for each topic.

        Args:
            topic_name: The topic to generate context for
            difficulty: FSRS difficulty (1-10)
            mastery_level: BKT mastery probability (0-1)

        Returns:
            A transcript-like text for the topic
        """
        # Map mastery to difficulty description
        if mastery_level < 0.3:
            level_desc = "introductory and fundamental"
        elif mastery_level < 0.6:
            level_desc = "intermediate with practical applications"
        elif mastery_level < 0.85:
            level_desc = "advanced with complex scenarios"
        else:
            level_desc = "expert-level with edge cases and optimization"

        # Create a rich context for the topic
        # In a real scenario, you might have actual transcripts stored
        context_templates = {
            "Python Variables & Data Types": f"""
                Python variables are fundamental building blocks that store data in memory.
                Understanding data types is essential for writing efficient code. Python 
                supports several built-in data types including integers, floats, strings, 
                booleans, lists, tuples, dictionaries, and sets. Variables in Python are 
                dynamically typed, meaning you don't need to declare their type explicitly.
                Type conversion, type checking with type() and isinstance(), and understanding
                mutability vs immutability are {level_desc} concepts. Common operations include
                string formatting, numeric operations, type casting, and working with None values.
                """,

            "Python Functions & Parameters": f"""
                Functions in Python are reusable blocks of code that perform specific tasks.
                They can accept parameters, return values, and have local scope. Understanding
                function definitions with def, positional vs keyword arguments, default parameters,
                *args and **kwargs, lambda functions, and decorators are {level_desc} topics.
                Functions promote code reusability and modularity. Key concepts include scope
                (local, global, nonlocal), return statements, docstrings, type hints, and 
                function composition.
                """,

            "Python Lists & Tuples": f"""
                Lists and tuples are sequence data types in Python. Lists are mutable ordered
                collections that can store elements of different types. Tuples are immutable
                sequences often used for fixed collections. Understanding list comprehensions,
                slicing, indexing, common list methods (append, extend, insert, remove, pop),
                sorting, and tuple packing/unpacking are {level_desc} concepts. Performance
                considerations, when to use lists vs tuples, and nested sequences are important.
                """,

            "Python Dictionaries": f"""
                Dictionaries are key-value pair data structures in Python that provide fast
                lookups. They're mutable, unordered (Python 3.7+ maintains insertion order),
                and incredibly versatile. Understanding dictionary creation, accessing values,
                adding/updating/deleting items, dictionary methods (get, keys, values, items,
                update, pop), dictionary comprehensions, and nested dictionaries are {level_desc}
                topics. Hash tables, time complexity, and use cases for dictionaries are key.
                """,

            "Python Loops (for/while)": f"""
                Loops allow you to execute code repeatedly. Python has two main loop types:
                for loops that iterate over sequences, and while loops that continue based on
                conditions. Understanding iteration, the range() function, enumerate(), zip(),
                break and continue statements, else clauses in loops, and nested loops are
                {level_desc} concepts. Loop optimization, avoiding infinite loops, and choosing
                the right loop type are important skills.
                """,

            "Python Conditionals (if/else)": f"""
                Conditional statements control program flow based on boolean expressions.
                Understanding if, elif, else syntax, comparison operators, logical operators
                (and, or, not), truthiness and falsiness, ternary operators, and nested
                conditionals are {level_desc} topics. Boolean logic, short-circuit evaluation,
                and writing clean conditional code are essential skills.
                """,

            "Python Classes & Objects": f"""
                Object-oriented programming in Python uses classes to create custom data types.
                Classes encapsulate data (attributes) and behavior (methods). Understanding
                class definition, __init__ constructors, self parameter, instance vs class
                attributes, instance vs class methods, inheritance, polymorphism, encapsulation,
                special methods (__str__, __repr__, etc.), and property decorators are {level_desc}
                topics. Design patterns and when to use OOP are important considerations.
                """,

            "Python Error Handling": f"""
                Error handling in Python uses try-except blocks to gracefully handle exceptions.
                Understanding different exception types (ValueError, TypeError, KeyError, etc.),
                try-except-else-finally blocks, raising exceptions, custom exceptions, and
                exception hierarchies are {level_desc} concepts. Best practices include catching
                specific exceptions, avoiding bare except clauses, using context managers, and
                writing defensive code.
                """
        }

        # Get context or use a generic template
        context = context_templates.get(
            topic_name,
            f"""
            {topic_name} is an important concept in programming. This topic covers
            {level_desc} material that builds on fundamental principles. Understanding
            the core concepts, practical applications, common patterns, and best practices
            are essential for mastery. Real-world examples demonstrate how these concepts
            are applied in software development.
            """
        )

        return context.strip()

    def generate_question(
            self,
            topic_name: str,
            difficulty: float,
            mastery_level: float,
            previous_questions: List[str] = None
    ) -> Dict:
        """
        Generate a question for a specific topic using Gemini.

        Args:
            topic_name: The topic to generate a question for
            difficulty: FSRS difficulty (1-10)
            mastery_level: BKT mastery probability (0-1)
            previous_questions: List of previous questions (not used with Gemini batch)

        Returns:
            Dictionary with question, options, correct_answer, and explanation
        """
        # Check if we have cached questions for this topic
        if topic_name not in self.quiz_cache or not self.quiz_cache[topic_name]:
            # Generate a new batch of 10 questions
            print(f"🤖 Generating new question set for: {topic_name}...")

            # Create topic-specific context
            transcript = self.generate_topic_context(topic_name, difficulty, mastery_level)

            try:
                # Generate 10 questions using your quiz_generator
                questions = generate_quiz(
                    transcript=transcript,
                    api_key=self.api_key
                )

                # Cache the questions
                self.quiz_cache[topic_name] = questions
                self.current_question_index[topic_name] = 0

                print(f"✅ Generated {len(questions)} questions for {topic_name}")

            except Exception as e:
                print(f"⚠️  Gemini generation error: {e}")
                # Fallback question
                return {
                    "question": f"What is a key concept in {topic_name}?",
                    "options": {
                        "A": "Concept A",
                        "B": "Concept B",
                        "C": "Concept C",
                        "D": "Concept D"
                    },
                    "correct_answer": "A",
                    "explanation": "Fallback question due to generation error."
                }

        # Get the next question from cache
        questions = self.quiz_cache[topic_name]
        index = self.current_question_index[topic_name]

        # Cycle through questions
        if index >= len(questions):
            index = 0
            self.current_question_index[topic_name] = 0

        question_data = questions[index]
        self.current_question_index[topic_name] = index + 1

        return question_data


class InteractiveQuiz:
    """Interactive quiz session with FSRS scheduling and Gemini questions"""

    def __init__(self, quiz_system: AdaptiveQuizSystem, generator: GeminiQuestionGenerator):
        self.quiz = quiz_system
        self.generator = generator
        self.history: Dict[str, List[str]] = {}

    def ask_question(self) -> Optional[Dict]:
        """Ask one question using FSRS scheduling"""

        # FSRS selects the next topic (due + UCB exploration)
        topic_id = self.quiz.select_next_topic()

        if not topic_id:
            print("\n✅ All topics reviewed! No topics due right now.")
            return None

        topic = self.quiz.topics[topic_id]

        # Generate question
        print(f"\n{'─' * 70}")
        print(f"📚 Topic: {topic.name}")
        print(
            f"📊 Mastery: {topic.bkt.p_mastery:.0%} | Difficulty: {topic.fsrs.difficulty:.1f}/10 | Reviews: {topic.fsrs.reps}")
        print(f"{'─' * 70}\n")

        prev = self.history.get(topic_id, [])
        q_data = self.generator.generate_question(
            topic.name,
            topic.fsrs.difficulty,
            topic.bkt.p_mastery,
            prev
        )

        # Save to history
        if topic_id not in self.history:
            self.history[topic_id] = []
        self.history[topic_id].append(q_data['question'])

        # Display
        print(f"❓ {q_data['question']}\n")
        for key, val in q_data['options'].items():
            print(f"   {key}) {val}")

        # Get answer
        while True:
            answer = input("\n👉 Your answer (A/B/C/D) or 'quit': ").strip().upper()
            if answer == 'QUIT':
                return None
            if answer in ['A', 'B', 'C', 'D']:
                break
            print("   Please enter A, B, C, or D")

        # Check
        correct = (answer == q_data['correct_answer'])
        print()
        if correct:
            print("✅ Correct!")
        else:
            print(f"❌ Wrong. Correct answer: {q_data['correct_answer']}")
        print(f"\n💡 {q_data['explanation']}\n")

        # Get confidence
        print("How confident were you?")
        print("  1=Complete guess | 2=Not confident | 3=Somewhat | 4=Confident | 5=Certain")

        while True:
            try:
                conf = int(input("Confidence (1-5): ").strip())
                if 1 <= conf <= 5:
                    break
            except:
                pass
            print("   Enter 1-5")

        # Quality score for FSRS
        quality = min(conf + 1, 5) if correct else max(0, conf - 3)

        # Submit to quiz system (FSRS updates here!)
        result = self.quiz.submit_answer(topic_id, correct, quality)

        # Show FSRS update
        print(f"\n📈 Mastery: {result['old_mastery']:.0%} → {result['new_mastery']:.0%}")

        # Convert minutes to readable format
        mins = result['next_review_minutes']
        if mins < 60:
            review_time = f"{mins:.0f} minutes"
        elif mins < 1440:
            review_time = f"{mins / 60:.1f} hours"
        else:
            review_time = f"{mins / 1440:.1f} days"
        print(f"⏰ Next review: {review_time}")

        return {
            'topic': topic.name,
            'correct': correct,
            'quality': quality,
            'mastery_delta': result['new_mastery'] - result['old_mastery']
        }

    def run(self, num_questions: int = 10):
        """Run quiz session"""
        self.quiz.start_session()

        print("\n" + "=" * 70)
        print("🎯 ADAPTIVE QUIZ WITH FSRS + GEMINI")
        print("=" * 70)
        print(f"\n📚 Topics: {len(self.quiz.topics)}")
        print(f"🎲 Questions: {num_questions}\n")

        results = []

        for i in range(num_questions):
            print(f"\n{'=' * 70}")
            print(f"QUESTION {i + 1}/{num_questions}")
            print(f"{'=' * 70}")

            result = self.ask_question()
            if result is None:
                break

            results.append(result)

            if i < num_questions - 1:
                input("\n⏎ Press Enter for next question...")

        self.show_summary(results)

    def show_summary(self, results: List[Dict]):
        """Show session summary"""
        if not results:
            return

        print("\n" + "=" * 70)
        print("📊 SESSION SUMMARY")
        print("=" * 70)

        correct = sum(1 for r in results if r['correct'])
        total = len(results)
        avg_quality = sum(r['quality'] for r in results) / total

        print(f"\n✅ Score: {correct}/{total} ({correct / total:.0%})")
        print(f"⭐ Avg Quality: {avg_quality:.1f}/5")

        # Per-topic breakdown
        print("\n📚 By Topic:")
        topics = {}
        for r in results:
            t = r['topic']
            if t not in topics:
                topics[t] = {'correct': 0, 'total': 0, 'delta': 0}
            topics[t]['total'] += 1
            if r['correct']:
                topics[t]['correct'] += 1
            topics[t]['delta'] += r['mastery_delta']

        for topic, stats in topics.items():
            acc = stats['correct'] / stats['total']
            print(f"  • {topic}: {stats['correct']}/{stats['total']} ({acc:.0%}) | Δ{stats['delta']:+.0%}")

        # Overall progress
        summary = self.quiz.get_session_summary()
        print(f"\n🎯 Overall Progress:")
        print(f"  Mastered: {summary['topics_mastered']}/{summary['topics_total']}")
        print(f"  Avg Mastery: {summary['average_mastery']:.0%}")
        print(f"  Duration: {summary['session_duration_minutes']:.1f} min")
        print("\n" + "=" * 70 + "\n")


def check_environment() -> bool:
    """Check if all required environment variables are set"""
    print("🔍 Checking environment configuration...\n")

    required_vars = {
        'SUPABASE_URL': 'Supabase project URL',
        'SUPABASE_KEY': 'Supabase anon/service key',
        'GEMINI_API_KEY': 'Google Gemini API key for question generation'
    }

    missing = []
    for var, description in required_vars.items():
        value = os.getenv(var)
        if value:
            # Mask sensitive parts
            if 'KEY' in var:
                display_value = f"{value[:20]}...{value[-4:]}" if len(value) > 24 else "***"
            else:
                display_value = value
            print(f"  ✅ {var}: {display_value}")
        else:
            print(f"  ❌ {var}: Not set ({description})")
            missing.append(var)

    print()

    if missing:
        print("❌ Missing required environment variables!")
        print("\n📝 Add to your .env file:")
        print("\n" + "─" * 70)
        for var in missing:
            print(f"{var}=your-value-here")
        print("─" * 70)
        print()
        return False

    print("✅ All environment variables are set!\n")
    return True


def main():
    """Main entry point"""
    print("=" * 70)
    print("🚀 ADAPTIVE QUIZ SYSTEM WITH GEMINI + FSRS")
    print("=" * 70)
    print()

    # Check environment
    if not check_environment():
        sys.exit(1)

    try:
        print("📡 Initializing quiz system...")

        # Initialize quiz system (credentials from .env)
        quiz = initialize_quiz_system(
            username="student_demo",
            topic_names=[
                "Python Variables & Data Types",
                "Python Functions & Parameters",
                "Python Lists & Tuples",
                "Python Dictionaries",
                "Python Loops (for/while)",
                "Python Conditionals (if/else)",
                "Python Classes & Objects",
                "Python Error Handling"
            ]
        )

        print("✅ Quiz system initialized!")
        print()

        # Initialize Gemini question generator (credentials from .env)
        print("🤖 Initializing Gemini question generator...")
        generator = GeminiQuestionGenerator()
        print("✅ Gemini generator ready!")
        print()

        # Create and run quiz
        interactive = InteractiveQuiz(quiz, generator)
        interactive.run(num_questions=8)

    except KeyboardInterrupt:
        print("\n\n👋 Quiz interrupted. Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()