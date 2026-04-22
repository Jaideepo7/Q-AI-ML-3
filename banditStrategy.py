"""
Thompson Sampling Multi-Armed Bandit for Adaptive Quiz Strategy Selection

Arms (Strategies):
1. Focus on weak topics (targets lowest mastery scores)
2. Mixed difficulty (balanced easy/medium/hard questions)
3. Spaced repetition priority (questions due for review)
4. Random exploration (diverse topic coverage)

Reward: Normalized user performance improvement (0.0 to 1.0)
"""

import numpy as np
from typing import Dict, List, Tuple
import json
from datetime import datetime


class QuizBandit:
    """
    Thompson Sampling bandit for selecting quiz generation strategies.
    Uses Beta distribution for each arm's reward probability.
    """
    
    def __init__(self, n_arms: int = 4, strategy_names: List[str] = None):
        """
        Initialize bandit with uniform prior (Beta(1,1) for each arm).
        
        Args:
            n_arms: Number of quiz strategies (default 4)
            strategy_names: Optional list of strategy names for logging
        """
        self.n_arms = n_arms
        
        # Default strategy names
        if strategy_names is None:
            self.strategy_names = [
                "focus_weak_topics",
                "mixed_difficulty", 
                "spaced_repetition_priority",
                "random_exploration"
            ]
        else:
            self.strategy_names = strategy_names
            
        # Beta distribution parameters (alpha, beta) for each arm
        # Start with uniform prior: Beta(1, 1)
        self.alphas = np.ones(n_arms)
        self.betas = np.ones(n_arms)
        
        # Track total pulls per arm for monitoring
        self.pulls = np.zeros(n_arms)
        
        # Track cumulative reward per arm
        self.cumulative_rewards = np.zeros(n_arms)
        
    def select_strategy(self, user_context: Dict = None) -> Tuple[int, str]:
        """
        Sample from each arm's posterior distribution and select the best.
        
        Args:
            user_context: Optional dict with user info (for future context-aware bandits)
                         e.g., {"user_id": "123", "session_count": 5}
        
        Returns:
            (strategy_id, strategy_name) tuple
        """
        # Sample from each arm's Beta distribution
        samples = np.random.beta(self.alphas, self.betas)
        
        # Select arm with highest sampled value
        selected_arm = np.argmax(samples)
        
        # Increment pull counter
        self.pulls[selected_arm] += 1
        
        return selected_arm, self.strategy_names[selected_arm]
    
    def update(self, strategy_id: int, reward: float):
        """
        Update the selected arm's Beta distribution based on observed reward.
        
        Args:
            strategy_id: Index of the strategy that was used
            reward: Performance score (0.0 to 1.0)
                   - 1.0 = perfect quiz performance
                   - 0.5 = 50% correct
                   - 0.0 = all wrong
        """
        # Validate inputs
        assert 0 <= strategy_id < self.n_arms, f"Invalid strategy_id: {strategy_id}"
        assert 0.0 <= reward <= 1.0, f"Reward must be in [0,1], got {reward}"
        
        # Bernoulli interpretation: treat reward as success probability
        # For Beta-Bernoulli conjugate prior:
        # - If reward >= 0.5, count as success (increment alpha)
        # - If reward < 0.5, count as failure (increment beta)
        
        # Alternative: Fractional update (smoother learning)
        # Increment alpha by reward, beta by (1 - reward)
        self.alphas[strategy_id] += reward
        self.betas[strategy_id] += (1.0 - reward)
        
        # Track cumulative reward
        self.cumulative_rewards[strategy_id] += reward
        
    def get_statistics(self) -> Dict:
        """
        Return current bandit statistics for monitoring/debugging.
        
        Returns:
            Dict with strategy performance metrics
        """
        stats = {
            "timestamp": datetime.now().isoformat(),
            "strategies": []
        }
        
        for i in range(self.n_arms):
            # Expected value of Beta(alpha, beta) = alpha / (alpha + beta)
            expected_reward = self.alphas[i] / (self.alphas[i] + self.betas[i])
            
            # Average observed reward
            avg_reward = (self.cumulative_rewards[i] / self.pulls[i] 
                         if self.pulls[i] > 0 else 0.0)
            
            stats["strategies"].append({
                "id": i,
                "name": self.strategy_names[i],
                "alpha": float(self.alphas[i]),
                "beta": float(self.betas[i]),
                "expected_reward": float(expected_reward),
                "total_pulls": int(self.pulls[i]),
                "cumulative_reward": float(self.cumulative_rewards[i]),
                "avg_observed_reward": float(avg_reward)
            })
            
        return stats
    
    def save_state(self, filepath: str):
        """Save bandit state to JSON file."""
        state = {
            "n_arms": self.n_arms,
            "strategy_names": self.strategy_names,
            "alphas": self.alphas.tolist(),
            "betas": self.betas.tolist(),
            "pulls": self.pulls.tolist(),
            "cumulative_rewards": self.cumulative_rewards.tolist()
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
    
    def load_state(self, filepath: str):
        """Load bandit state from JSON file."""
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        self.n_arms = state["n_arms"]
        self.strategy_names = state["strategy_names"]
        self.alphas = np.array(state["alphas"])
        self.betas = np.array(state["betas"])
        self.pulls = np.array(state["pulls"])
        self.cumulative_rewards = np.array(state["cumulative_rewards"])


# ============================================================================
# Example Usage & Testing
# ============================================================================

def simulate_quiz_session(bandit: QuizBandit, true_strategy_rewards: np.ndarray) -> float:
    """
    Simulate one quiz session with the bandit.
    
    Args:
        bandit: QuizBandit instance
        true_strategy_rewards: Ground truth reward for each strategy
    
    Returns:
        Observed reward
    """
    # Select strategy
    strategy_id, strategy_name = bandit.select_strategy()
    
    # Simulate noisy reward (true reward + Gaussian noise)
    true_reward = true_strategy_rewards[strategy_id]
    noise = np.random.normal(0, 0.1)  # Small noise
    observed_reward = np.clip(true_reward + noise, 0.0, 1.0)
    
    # Update bandit
    bandit.update(strategy_id, observed_reward)
    
    return observed_reward


def main():
    """Demo: Test bandit with simulated quiz sessions."""
    
    print("=" * 60)
    print("Thompson Sampling Bandit Demo")
    print("=" * 60)
    
    # Initialize bandit
    bandit = QuizBandit(n_arms=4)
    
    # Ground truth: Strategy 0 (focus_weak_topics) is best with 0.75 avg reward
    # Others are progressively worse
    true_rewards = np.array([0.75, 0.60, 0.55, 0.45])
    
    print("\nGround Truth Strategy Rewards:")
    for i, (name, reward) in enumerate(zip(bandit.strategy_names, true_rewards)):
        print(f"  {i}. {name}: {reward:.2f}")
    
    # Run 100 simulated quiz sessions
    n_sessions = 100
    print(f"\nRunning {n_sessions} simulated quiz sessions...\n")
    
    for session in range(n_sessions):
        reward = simulate_quiz_session(bandit, true_rewards)
        
        # Print progress every 20 sessions
        if (session + 1) % 20 == 0:
            print(f"Session {session + 1}/{n_sessions} complete")
    
    # Display final statistics
    print("\n" + "=" * 60)
    print("Final Bandit Statistics")
    print("=" * 60)
    
    stats = bandit.get_statistics()
    
    for strategy in stats["strategies"]:
        print(f"\n{strategy['name']}:")
        print(f"  Expected Reward: {strategy['expected_reward']:.3f}")
        print(f"  Pulls: {strategy['total_pulls']}")
        print(f"  Avg Observed Reward: {strategy['avg_observed_reward']:.3f}")
        print(f"  Beta Params: α={strategy['alpha']:.1f}, β={strategy['beta']:.1f}")
    
    # Check if bandit learned to prefer best strategy
    best_strategy_id = np.argmax([s['total_pulls'] for s in stats['strategies']])
    print(f"\nMost selected strategy: {stats['strategies'][best_strategy_id]['name']}")
    print(f"(Ground truth best: {bandit.strategy_names[0]})")
    
    # Save state
    bandit.save_state("/home/claude/bandit_state.json")
    print("\nBandit state saved to bandit_state.json")


if __name__ == "__main__":
    main()
