import unittest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.estate import (
    RentingScenario,
    BuyingScenario,
    evaluate_renting,
    evaluate_buying,
    compare_scenarios,
)

class TestRentingScenario(unittest.TestCase):
    """Test RentingScenario evaluation"""

    def test_evaluate_renting_basic(self):
        """Test basic renting scenario with simple values"""
        scenario = RentingScenario(
            rent_monthly=1000.0,
            utilities_monthly=200.0,
            living_months=12,
        )
        result = evaluate_renting(scenario)

        # Check basic calculations
        self.assertEqual(result["months"], 12)
        self.assertEqual(result["avg_net_cost_per_month"], 1200.0)
        self.assertEqual(result["total_spent"], 14400.0)  # 1200 * 12
        self.assertEqual(result["total_income"], 0.0)
        self.assertEqual(result["net_cashflow"], -14400.0)
        self.assertEqual(result["wealth_end"], -14400.0)
        self.assertEqual(len(result["monthly_net_cost"]), 12)
        self.assertTrue(all(cost == 1200.0 for cost in result["monthly_net_cost"]))

    def test_evaluate_renting_with_metadata(self):
        """Test that description and link are included when provided"""
        scenario = RentingScenario(
            rent_monthly=1000.0,
            utilities_monthly=200.0,
            living_months=6,
            description="Test apartment",
            link="https://example.com/apartment",
        )
        result = evaluate_renting(scenario)

        self.assertEqual(result["description"], "Test apartment")
        self.assertEqual(result["link"], "https://example.com/apartment")

    def test_evaluate_renting_without_metadata(self):
        """Test that metadata fields are not included when not provided"""
        scenario = RentingScenario(
            rent_monthly=1000.0,
            utilities_monthly=200.0,
            living_months=6,
        )
        result = evaluate_renting(scenario)

        self.assertNotIn("description", result)
        self.assertNotIn("link", result)


class TestBuyingScenario(unittest.TestCase):
    """Test BuyingScenario evaluation"""

    def setUp(self):
        """Create a simple buying scenario for testing"""
        self.simple_scenario = BuyingScenario(
            purchase_price=100000.0,
            mortgage_principal=80000.0,  # 20k down payment
            mortgage_annual_rate=0.0,  # 0% interest for simplicity
            mortgage_term_years=10,
            living_months=12,
            one_off_costs=5000.0,
            renovation_costs_once=0.0,
            monthly_vve=100.0,
            monthly_utilities=150.0,
            monthly_rent_income=0.0,
            months_rented=0,
            annual_value_growth=0.0,  # No growth for simplicity
            sold_at_end=False,
            selling_cost_rate=0.0,
        )

    def test_evaluate_buying_basic_no_interest(self):
        """Test buying scenario with 0% interest rate"""
        result = evaluate_buying(self.simple_scenario)

        # Check basic fields exist
        self.assertEqual(result["months"], 12)
        self.assertIn("total_spent", result)
        self.assertIn("total_income", result)
        self.assertIn("net_cashflow", result)
        self.assertIn("wealth_end", result)
        self.assertIn("home_value_end", result)

        # With 0% interest and 0% growth, home value should stay same
        self.assertAlmostEqual(result["home_value_end"], 100000.0, places=2)

        # Total income should be 0 (no rent)
        self.assertEqual(result["total_income"], 0.0)

        # Mortgage interest should be 0
        self.assertEqual(result["mortgage_interest_paid"], 0.0)

    def test_evaluate_buying_with_sale(self):
        """Test buying scenario when property is sold at end"""
        scenario = BuyingScenario(
            purchase_price=100000.0,
            mortgage_principal=80000.0,
            mortgage_annual_rate=0.0,
            mortgage_term_years=10,
            living_months=12,
            one_off_costs=5000.0,
            renovation_costs_once=0.0,
            monthly_vve=100.0,
            monthly_utilities=150.0,
            monthly_rent_income=0.0,
            months_rented=0,
            annual_value_growth=0.0,
            sold_at_end=True,
            selling_cost_rate=0.02,  # 2% selling costs
            selling_cost_fixed=1000.0,
        )
        result = evaluate_buying(scenario)

        # Check that selling costs are calculated
        expected_selling_costs = 100000.0 * 0.02 + 1000.0  # 3000
        self.assertAlmostEqual(result["selling_costs_if_sold"], expected_selling_costs, places=2)

        # Net sale proceeds should account for selling costs and remaining debt
        self.assertIn("net_sale_proceeds_if_sold", result)
        self.assertGreater(result["net_sale_proceeds_if_sold"], 0)

    def test_evaluate_buying_with_rent_income(self):
        """Test buying scenario with rental income"""
        scenario = BuyingScenario(
            purchase_price=100000.0,
            mortgage_principal=80000.0,
            mortgage_annual_rate=0.0,
            mortgage_term_years=10,
            living_months=12,
            one_off_costs=5000.0,
            renovation_costs_once=0.0,
            monthly_vve=100.0,
            monthly_utilities=150.0,
            monthly_rent_income=800.0,
            months_rented=6,  # Rented for 6 months
            annual_value_growth=0.0,
            sold_at_end=False,
            selling_cost_rate=0.0,
        )
        result = evaluate_buying(scenario)

        # Total income should be 6 * 800 = 4800
        self.assertEqual(result["total_income"], 4800.0)

    def test_evaluate_buying_with_metadata(self):
        """Test that description and link are included when provided"""
        scenario = BuyingScenario(
            purchase_price=100000.0,
            mortgage_principal=80000.0,
            mortgage_annual_rate=0.0,
            mortgage_term_years=10,
            living_months=12,
            one_off_costs=5000.0,
            renovation_costs_once=0.0,
            monthly_vve=100.0,
            monthly_utilities=150.0,
            monthly_rent_income=0.0,
            months_rented=0,
            annual_value_growth=0.0,
            sold_at_end=False,
            selling_cost_rate=0.0,
            description="Test property",
            link="https://example.com/property",
        )
        result = evaluate_buying(scenario)

        self.assertEqual(result["description"], "Test property")
        self.assertEqual(result["link"], "https://example.com/property")

    def test_evaluate_buying_monthly_net_cost_length(self):
        """Test that monthly_net_cost has correct length"""
        result = evaluate_buying(self.simple_scenario)

        self.assertEqual(len(result["monthly_net_cost"]), 12)
        # All values should be positive (costs)
        self.assertTrue(all(cost > 0 for cost in result["monthly_net_cost"]))

    def test_evaluate_buying_with_interest_rate(self):
        """Test buying scenario with non-zero interest rate"""
        scenario = BuyingScenario(
            purchase_price=100000.0,
            mortgage_principal=80000.0,
            mortgage_annual_rate=0.06,  # 6% annual = 0.5% monthly
            mortgage_term_years=10,
            living_months=12,
            one_off_costs=5000.0,
            renovation_costs_once=0.0,
            monthly_vve=100.0,
            monthly_utilities=150.0,
            monthly_rent_income=0.0,
            months_rented=0,
            annual_value_growth=0.0,
            sold_at_end=False,
            selling_cost_rate=0.0,
        )
        result = evaluate_buying(scenario)

        # Calculate expected values:
        # Monthly rate: r = 0.06 / 12 = 0.005
        # Monthly payment formula: P * [r(1+r)^n] / [(1+r)^n - 1]
        # P = 80000, r = 0.005, n = 120
        # Monthly payment = 888.164016
        expected_monthly_payment = 888.164016
        
        # After 12 months, calculate interest and principal month by month:
        # Month 1: interest = 80000 * 0.005 = 400.00, principal = 488.16, balance = 79511.84
        # Month 2: interest = 79511.84 * 0.005 = 397.56, principal = 490.60, balance = 79021.23
        # Month 3: interest = 79021.23 * 0.005 = 395.11, principal = 493.06, balance = 78528.17
        # ... continuing for 12 months
        # Total interest (12 months) = 4636.19
        # Total principal (12 months) = 6021.78
        # Remaining balance = 73978.22
        expected_interest_12mo = 4636.19
        expected_principal_12mo = 6021.78
        expected_remaining_balance = 73978.22
        
        self.assertAlmostEqual(result["mortgage_monthly_payment"], expected_monthly_payment, places=2)
        self.assertAlmostEqual(result["mortgage_interest_paid"], expected_interest_12mo, places=0)
        self.assertAlmostEqual(result["mortgage_principal_paid"], expected_principal_12mo, places=0)
        self.assertAlmostEqual(result["mortgage_remaining_balance"], expected_remaining_balance, places=0)

    def test_evaluate_buying_with_value_growth(self):
        """Test buying scenario with property value growth"""
        scenario = BuyingScenario(
            purchase_price=100000.0,
            mortgage_principal=80000.0,
            mortgage_annual_rate=0.0,
            mortgage_term_years=10,
            living_months=12,
            one_off_costs=5000.0,
            renovation_costs_once=0.0,
            monthly_vve=100.0,
            monthly_utilities=150.0,
            monthly_rent_income=0.0,
            months_rented=0,
            annual_value_growth=0.12,  # 12% annual growth
            sold_at_end=False,
            selling_cost_rate=0.0,
        )
        result = evaluate_buying(scenario)

        # Calculate expected home value after 12 months:
        # Monthly growth rate: (1.12)^(1/12) - 1 = 0.009488793
        # Future value: 100000 * (1.009488793)^12 = 100000 * 1.12 = 112,000
        expected_home_value = 112000.0
        
        # With 0% interest, after 12 months:
        # Principal paid = 80000 / 120 * 12 = 8000
        # Remaining debt = 80000 - 8000 = 72000
        # Equity = 112000 - 72000 = 40000
        expected_equity = 40000.0
        
        self.assertAlmostEqual(result["home_value_end"], expected_home_value, places=0)
        self.assertAlmostEqual(result["equity_end_if_not_sold"], expected_equity, places=0)
        self.assertAlmostEqual(result["mortgage_remaining_balance"], 72000.0, places=0)

    def test_evaluate_buying_cash_required_basic(self):
        """Test cash requirement calculation without renovation costs"""
        scenario = BuyingScenario(
            purchase_price=100000.0,
            mortgage_principal=80000.0,
            mortgage_annual_rate=0.0,
            mortgage_term_years=10,
            living_months=12,
            one_off_costs=5000.0,
            renovation_costs_once=0.0,
            monthly_vve=100.0,
            monthly_utilities=150.0,
            monthly_rent_income=0.0,
            months_rented=0,
            annual_value_growth=0.0,
            sold_at_end=False,
            selling_cost_rate=0.0,
        )
        result = evaluate_buying(scenario)
        
        # Down payment: 100000 - 80000 = 20000
        # Cash required: 20000 + 5000 + 0 = 25000
        self.assertEqual(result["down_payment"], 20000.0)
        self.assertEqual(result["cash_required_upfront"], 25000.0)
        self.assertEqual(result["one_off_costs"], 5000.0)
        self.assertEqual(result["renovation_costs_once"], 0.0)

    def test_evaluate_buying_cash_required_with_renovation(self):
        """Test cash requirement calculation with renovation costs"""
        scenario = BuyingScenario(
            purchase_price=200000.0,
            mortgage_principal=180000.0,
            mortgage_annual_rate=0.04,
            mortgage_term_years=20,
            living_months=60,
            one_off_costs=12000.0,
            renovation_costs_once=8000.0,
            monthly_vve=200.0,
            monthly_utilities=200.0,
            monthly_rent_income=0.0,
            months_rented=0,
            annual_value_growth=0.02,
            sold_at_end=True,
            selling_cost_rate=0.03,
        )
        result = evaluate_buying(scenario)
        
        # Down payment: 200000 - 180000 = 20000
        # Cash required: 20000 + 12000 + 8000 = 40000
        self.assertEqual(result["down_payment"], 20000.0)
        self.assertEqual(result["cash_required_upfront"], 40000.0)
        self.assertEqual(result["one_off_costs"], 12000.0)
        self.assertEqual(result["renovation_costs_once"], 8000.0)

    def test_evaluate_buying_cash_required_high_down_payment(self):
        """Test cash requirement with higher down payment (lower LTV)"""
        scenario = BuyingScenario(
            purchase_price=300000.0,
            mortgage_principal=240000.0,  # 80% LTV, 20% down
            mortgage_annual_rate=0.035,
            mortgage_term_years=25,
            living_months=120,
            one_off_costs=18000.0,
            renovation_costs_once=5000.0,
            monthly_vve=250.0,
            monthly_utilities=180.0,
            monthly_rent_income=0.0,
            months_rented=0,
            annual_value_growth=0.03,
            sold_at_end=False,
            selling_cost_rate=0.0,
        )
        result = evaluate_buying(scenario)
        
        # Down payment: 300000 - 240000 = 60000
        # Cash required: 60000 + 18000 + 5000 = 83000
        self.assertEqual(result["down_payment"], 60000.0)
        self.assertEqual(result["cash_required_upfront"], 83000.0)
        self.assertEqual(result["one_off_costs"], 18000.0)
        self.assertEqual(result["renovation_costs_once"], 5000.0)

    def test_evaluate_buying_usual_monthly_cost(self):
        """Test usual monthly cost calculation (excluding one-off costs)"""
        scenario = BuyingScenario(
            purchase_price=200000.0,
            mortgage_principal=180000.0,
            mortgage_annual_rate=0.04,
            mortgage_term_years=20,
            living_months=24,
            one_off_costs=12000.0,
            renovation_costs_once=8000.0,
            monthly_vve=200.0,
            monthly_utilities=150.0,
            monthly_rent_income=0.0,
            months_rented=0,
            annual_value_growth=0.0,
            sold_at_end=False,
            selling_cost_rate=0.0,
        )
        result = evaluate_buying(scenario)
        
        # Usual monthly cost should NOT include one-off or renovation costs
        # It should only include: mortgage payment + VVE + utilities
        # avg_net_cost_per_month INCLUDES one-off costs amortized
        # usual_net_cost_per_month EXCLUDES one-off costs
        
        self.assertIn("usual_net_cost_per_month", result)
        self.assertIn("avg_net_cost_per_month", result)
        
        # usual_net_cost_per_month should be less than avg_net_cost_per_month
        # because it excludes the 20,000 in one-off costs
        self.assertLess(
            result["usual_net_cost_per_month"],
            result["avg_net_cost_per_month"]
        )
        
        # The difference should be approximately (one_off + renovation) / months
        # = (12000 + 8000) / 24 = 833.33
        expected_difference = (12000.0 + 8000.0) / 24
        actual_difference = result["avg_net_cost_per_month"] - result["usual_net_cost_per_month"]
        self.assertAlmostEqual(actual_difference, expected_difference, places=0)

    def test_evaluate_buying_usual_monthly_cost_with_rent_income(self):
        """Test usual monthly cost with rental income"""
        scenario = BuyingScenario(
            purchase_price=200000.0,
            mortgage_principal=180000.0,
            mortgage_annual_rate=0.04,
            mortgage_term_years=20,
            living_months=12,
            one_off_costs=10000.0,
            renovation_costs_once=5000.0,
            monthly_vve=200.0,
            monthly_utilities=150.0,
            monthly_rent_income=800.0,
            months_rented=12,
            annual_value_growth=0.0,
            sold_at_end=False,
            selling_cost_rate=0.0,
        )
        result = evaluate_buying(scenario)
        
        # With rental income, usual monthly cost should be reduced
        # Both metrics should account for rental income
        self.assertIn("usual_net_cost_per_month", result)
        
        # The difference between avg and usual should still be the one-off costs
        expected_difference = (10000.0 + 5000.0) / 12
        actual_difference = result["avg_net_cost_per_month"] - result["usual_net_cost_per_month"]
        self.assertAlmostEqual(actual_difference, expected_difference, places=0)


class TestCompareScenarios(unittest.TestCase):
    """Test compare_scenarios function"""

    def test_compare_both_scenarios(self):
        """Test comparing both renting and buying scenarios"""
        rent = RentingScenario(
            rent_monthly=1000.0,
            utilities_monthly=200.0,
            living_months=12,
        )
        buy = BuyingScenario(
            purchase_price=100000.0,
            mortgage_principal=80000.0,
            mortgage_annual_rate=0.0,
            mortgage_term_years=10,
            living_months=12,
            one_off_costs=5000.0,
            renovation_costs_once=0.0,
            monthly_vve=100.0,
            monthly_utilities=150.0,
            monthly_rent_income=0.0,
            months_rented=0,
            annual_value_growth=0.0,
            sold_at_end=False,
            selling_cost_rate=0.0,
        )

        result = compare_scenarios(rent, buy)

        # Should have all three keys
        self.assertIn("renting", result)
        self.assertIn("buying", result)
        self.assertIn("invest", result)

        # Check invest result structure
        self.assertIn("renting_invests", result["invest"])
        self.assertIn("buying_invests", result["invest"])
        self.assertIn("principal_r", result["invest"])
        self.assertIn("principal_b", result["invest"])

    def test_compare_only_renting(self):
        """Test comparing with only renting scenario"""
        rent = RentingScenario(
            rent_monthly=1000.0,
            utilities_monthly=200.0,
            living_months=12,
        )

        result = compare_scenarios(rent, None)

        self.assertIn("renting", result)
        self.assertNotIn("buying", result)
        self.assertNotIn("invest", result)

    def test_compare_only_buying(self):
        """Test comparing with only buying scenario"""
        buy = BuyingScenario(
            purchase_price=100000.0,
            mortgage_principal=80000.0,
            mortgage_annual_rate=0.0,
            mortgage_term_years=10,
            living_months=12,
            one_off_costs=5000.0,
            renovation_costs_once=0.0,
            monthly_vve=100.0,
            monthly_utilities=150.0,
            monthly_rent_income=0.0,
            months_rented=0,
            annual_value_growth=0.0,
            sold_at_end=False,
            selling_cost_rate=0.0,
        )

        result = compare_scenarios(None, buy)

        self.assertNotIn("renting", result)
        self.assertIn("buying", result)
        self.assertNotIn("invest", result)


if __name__ == "__main__":
    unittest.main()
