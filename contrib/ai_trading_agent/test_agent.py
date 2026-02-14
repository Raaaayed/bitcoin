import unittest

from risk_managed_agent import AgentConfig, RiskManagedAgent, TradeSignal, score_strategy


class RiskManagedAgentTests(unittest.TestCase):
    def test_position_respects_rr_ratio(self):
        agent = RiskManagedAgent(AgentConfig(reward_to_risk=2.0))
        plan = agent.plan_position(TradeSignal(0.8, 1), price=100.0, capital=10_000.0)
        self.assertIsNotNone(plan)
        assert plan is not None
        risk = plan.entry - plan.stop_loss
        reward = plan.take_profit - plan.entry
        self.assertAlmostEqual(reward / risk, 2.0)

    def test_low_confidence_blocks_trade(self):
        agent = RiskManagedAgent()
        plan = agent.plan_position(TradeSignal(0.2, 1), price=100.0, capital=10_000.0)
        self.assertIsNone(plan)

    def test_score_targets(self):
        report = score_strategy([120, -40, 140, -30, 130], starting_capital=10_000.0)
        self.assertGreater(report["win_rate"], 0.5)
        self.assertGreaterEqual(report["risk_reward"], 2.0)


if __name__ == "__main__":
    unittest.main()
