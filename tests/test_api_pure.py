"""Tests for ShoppingListWithGrocyApi pure methods.

These tests cover methods that don't require a live HA instance or HTTP calls.
The API object is instantiated with a minimal stub config and a mock hass.
"""

from unittest.mock import MagicMock

import pytest

# ── Helpers ──────────────────────────────────────────────────────────────────


def make_api(image_size=0, bidirectional=False):
    """Return an API instance with a minimal stub config, no real HTTP session."""
    from custom_components.shopping_list_with_grocy.apis.shopping_list_with_grocy import (
        ShoppingListWithGrocyApi,
    )

    hass = MagicMock()
    hass.config.language = "en"
    hass.data = {}

    session = MagicMock()
    config = {
        "api_url": "http://grocy.local",
        "api_key": "test-key",
        "image_download_size": image_size,
        "disable_timeout": False,
        "enable_bidirectional_sync": bidirectional,
    }
    api = ShoppingListWithGrocyApi(session, hass, config)
    return api


# ── encode_base64 ─────────────────────────────────────────────────────────────


class TestEncodeBase64:
    def test_simple_string(self):
        api = make_api()
        import base64

        assert api.encode_base64("hello") == base64.b64encode(b"hello").decode()

    def test_empty_string(self):
        api = make_api()
        assert api.encode_base64("") == ""

    def test_non_string_raises(self):
        api = make_api()
        with pytest.raises(TypeError):
            api.encode_base64(123)


# ── normalize_text_for_search ─────────────────────────────────────────────────


class TestNormalizeTextForSearch:
    def test_removes_accents(self):
        api = make_api()
        assert api.normalize_text_for_search("Pâtes") == "pates"

    def test_lowercases(self):
        api = make_api()
        assert api.normalize_text_for_search("LAIT") == "lait"

    def test_strips_whitespace(self):
        api = make_api()
        assert api.normalize_text_for_search("  beurre  ") == "beurre"

    def test_empty_string(self):
        api = make_api()
        assert api.normalize_text_for_search("") == ""

    def test_none_returns_empty(self):
        api = make_api()
        assert api.normalize_text_for_search(None) == ""

    def test_complex_accents(self):
        api = make_api()
        assert api.normalize_text_for_search("Crème fraîche") == "creme fraiche"


# ── calculate_similarity ──────────────────────────────────────────────────────


class TestCalculateSimilarity:
    def test_identical_strings(self):
        api = make_api()
        assert api.calculate_similarity("lait", "lait") == 1.0

    def test_completely_different(self):
        api = make_api()
        score = api.calculate_similarity("lait", "xyzxyz")
        assert score < 0.3

    def test_partial_match(self):
        api = make_api()
        score = api.calculate_similarity("lait", "lait entier")
        assert 0.5 < score < 1.0

    def test_case_insensitive(self):
        api = make_api()
        assert api.calculate_similarity("LAIT", "lait") == 1.0

    def test_accent_insensitive(self):
        api = make_api()
        score = api.calculate_similarity("pates", "Pâtes")
        assert score == 1.0

    def test_empty_strings_return_zero(self):
        api = make_api()
        assert api.calculate_similarity("", "lait") == 0.0
        assert api.calculate_similarity("lait", "") == 0.0


# ── is_case_only_difference ───────────────────────────────────────────────────


class TestIsCaseOnlyDifference:
    def test_case_difference(self):
        api = make_api()
        assert api.is_case_only_difference("lait", "Lait") is True

    def test_same_string(self):
        api = make_api()
        assert api.is_case_only_difference("lait", "lait") is False

    def test_different_content(self):
        api = make_api()
        assert api.is_case_only_difference("lait", "beurre") is False


# ── extract_product_name_from_ha_item ─────────────────────────────────────────


class TestExtractProductName:
    def test_name_with_quantity_pattern1(self):
        """'Lait (x3)' → ('Lait', 3)"""
        api = make_api()
        name, qty = api.extract_product_name_from_ha_item("Lait (x3)")
        assert name == "Lait"
        assert qty == 3

    def test_name_with_quantity_pattern1_unicode_times(self):
        """'Beurre (×2)' → ('Beurre', 2)"""
        api = make_api()
        name, qty = api.extract_product_name_from_ha_item("Beurre (×2)")
        assert name == "Beurre"
        assert qty == 2

    def test_name_with_leading_number_pattern2(self):
        """'3 Lait' → ('Lait', 3)"""
        api = make_api()
        name, qty = api.extract_product_name_from_ha_item("3 Lait")
        assert name == "Lait"
        assert qty == 3

    def test_plain_name_no_quantity(self):
        """'Lait' → ('Lait', 1)"""
        api = make_api()
        name, qty = api.extract_product_name_from_ha_item("Lait")
        assert name == "Lait"
        assert qty == 1

    def test_strips_whitespace(self):
        api = make_api()
        name, qty = api.extract_product_name_from_ha_item("  Lait  ")
        assert name == "Lait"
        assert qty == 1

    def test_multiword_name(self):
        api = make_api()
        name, qty = api.extract_product_name_from_ha_item("Crème fraîche (x1)")
        assert name == "Crème fraîche"
        assert qty == 1


# ── compute_timeout ───────────────────────────────────────────────────────────


class TestComputeTimeout:
    @pytest.mark.parametrize(
        "image_size,expected",
        [
            (0, 60),
            (50, 60),
            (100, 90),
            (150, 120),
            (200, 180),
        ],
    )
    def test_known_sizes(self, image_size, expected):
        api = make_api(image_size=image_size)
        assert api.compute_timeout() == expected

    def test_unknown_size_picks_nearest(self):
        """image_size=75 → nearest key is 50 → timeout 60."""
        api = make_api(image_size=75)
        assert api.compute_timeout() == 60


# ── build_item_list — note-only items (issue #73 regression) ──────────────────


class TestBuildItemList:
    """Regression tests for issue #73: Grocy items with product_id=None crash."""

    def _make_data(self, shopping_list_items):
        return {
            "shopping_lists": [{"id": 1, "name": "Liste principale"}],
            "products": [
                {
                    "id": "1",
                    "name": "Lait",
                    "qu_id_purchase": "1",
                    "qu_id_stock": "1",
                    "qu_factor_purchase_to_stock": 1.0,
                },
            ],
            "shopping_list": shopping_list_items,
        }

    def test_normal_item(self):
        api = make_api()
        data = self._make_data(
            [
                {
                    "id": "10",
                    "product_id": "1",
                    "shopping_list_id": 1,
                    "amount": 2,
                    "done": 0,
                },
            ]
        )
        result = api.build_item_list(data)
        assert len(result) == 1
        assert len(result[0]["products"]) == 1
        assert "Lait" in result[0]["products"][0]["name"]

    def test_note_only_item_does_not_crash(self):
        """A shopping list entry with product_id=None must not raise TypeError."""
        api = make_api()
        data = self._make_data(
            [
                {
                    "id": "99",
                    "product_id": None,
                    "shopping_list_id": 1,
                    "amount": 1,
                    "done": 0,
                },
            ]
        )
        # Should not raise
        result = api.build_item_list(data)
        assert isinstance(result, list)
        # The note-only item has no product match → list is empty
        assert result[0]["products"] == []

    def test_mixed_normal_and_note_only(self):
        """Normal items are listed; note-only items are silently skipped."""
        api = make_api()
        data = self._make_data(
            [
                {
                    "id": "10",
                    "product_id": "1",
                    "shopping_list_id": 1,
                    "amount": 1,
                    "done": 0,
                },
                {
                    "id": "99",
                    "product_id": None,
                    "shopping_list_id": 1,
                    "amount": 1,
                    "done": 0,
                },
            ]
        )
        result = api.build_item_list(data)
        assert len(result[0]["products"]) == 1


# ── find_similar_products ─────────────────────────────────────────────────────


class TestFindSimilarProducts:
    def _setup_api_with_products(self, products):
        api = make_api()
        api.final_data = {
            "products": [{"id": str(i), "name": p} for i, p in enumerate(products)]
        }
        return api

    def test_finds_exact_match(self):
        api = self._setup_api_with_products(["Lait", "Beurre", "Fromage"])
        results = api.find_similar_products("Lait", threshold=0.8)
        assert len(results) >= 1
        assert results[0]["name"] == "Lait"

    def test_finds_fuzzy_match(self):
        api = self._setup_api_with_products(["Lait entier", "Beurre doux"])
        results = api.find_similar_products("lait", threshold=0.5)
        names = [r["name"] for r in results]
        assert "Lait entier" in names

    def test_no_match_returns_empty(self):
        api = self._setup_api_with_products(["Lait", "Beurre"])
        results = api.find_similar_products("xyzxyz", threshold=0.9)
        assert results == []

    def test_results_sorted_by_similarity(self):
        api = self._setup_api_with_products(["Lait entier", "Lait"])
        results = api.find_similar_products("Lait", threshold=0.5)
        scores = [r["similarity"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_empty_search_returns_empty(self):
        api = self._setup_api_with_products(["Lait"])
        assert api.find_similar_products("") == []

    def test_no_products_returns_empty(self):
        api = make_api()
        api.final_data = {}
        assert api.find_similar_products("Lait") == []


# ── apply_quantity_unit_conversions (issue #76) ───────────────────────────────

# Quantity unit ids used across the conversion tests.
QU_BOTTLE = 1
QU_BOX = 2
QU_PACK = 3


def make_product(product_id=1, qu_purchase=QU_PACK, qu_stock=QU_BOTTLE):
    """Return a Grocy 4 style product, without qu_factor_purchase_to_stock."""
    return {
        "id": str(product_id),
        "name": f"Product {product_id}",
        "qu_id_purchase": qu_purchase,
        "qu_id_stock": qu_stock,
    }


def make_conversion(from_qu, to_qu, factor, product_id=None):
    """Return a row as returned by /api/objects/quantity_unit_conversions."""
    return {
        "id": 1,
        "from_qu_id": from_qu,
        "to_qu_id": to_qu,
        "factor": factor,
        "product_id": product_id,
    }


class TestApplyQuantityUnitConversions:
    def test_direct_product_specific_conversion(self):
        """1 Pack = 18 Bottles, defined directly on the product."""
        api = make_api()
        products = [make_product()]
        conversions = [
            make_conversion(QU_PACK, QU_BOTTLE, 18, product_id=1),
            make_conversion(QU_BOTTLE, QU_PACK, 1 / 18, product_id=1),
        ]

        api.apply_quantity_unit_conversions(products, conversions)

        assert products[0]["qu_factor_purchase_to_stock"] == 18.0

    def test_transitive_conversion_chain(self):
        """Pack to Box to Bottle resolves to the product of both factors."""
        api = make_api()
        products = [make_product()]
        conversions = [
            make_conversion(QU_PACK, QU_BOX, 4),
            make_conversion(QU_BOX, QU_BOTTLE, 6),
        ]

        api.apply_quantity_unit_conversions(products, conversions)

        assert products[0]["qu_factor_purchase_to_stock"] == 24.0

    def test_inverse_conversion_is_used_when_direct_is_missing(self):
        """Only the stock to purchase direction exists, so it gets inverted."""
        api = make_api()
        products = [make_product()]
        conversions = [make_conversion(QU_BOTTLE, QU_PACK, 0.5, product_id=1)]

        api.apply_quantity_unit_conversions(products, conversions)

        assert products[0]["qu_factor_purchase_to_stock"] == 2.0

    def test_product_specific_wins_over_default(self):
        """A product override replaces the default conversion for the same pair."""
        api = make_api()
        products = [make_product()]
        conversions = [
            make_conversion(QU_PACK, QU_BOTTLE, 12),
            make_conversion(QU_PACK, QU_BOTTLE, 18, product_id=1),
        ]

        api.apply_quantity_unit_conversions(products, conversions)

        assert products[0]["qu_factor_purchase_to_stock"] == 18.0

    def test_default_still_applies_to_other_products(self):
        """An override on one product must not leak onto another one."""
        api = make_api()
        products = [make_product(product_id=1), make_product(product_id=2)]
        conversions = [
            make_conversion(QU_PACK, QU_BOTTLE, 12),
            make_conversion(QU_PACK, QU_BOTTLE, 18, product_id=1),
        ]

        api.apply_quantity_unit_conversions(products, conversions)

        assert products[0]["qu_factor_purchase_to_stock"] == 18.0
        assert products[1]["qu_factor_purchase_to_stock"] == 12.0

    def test_unavailable_endpoint_does_not_break_sync(self):
        """A failed fetch lands an exception in final_data instead of a list."""
        api = make_api()
        products = [make_product()]

        api.apply_quantity_unit_conversions(products, Exception("404 Not Found"))

        assert "qu_factor_purchase_to_stock" not in products[0]

    @pytest.mark.parametrize("conversions", [None, [], {}, "oops"])
    def test_unusable_conversions_are_ignored(self, conversions):
        api = make_api()
        products = [make_product()]

        api.apply_quantity_unit_conversions(products, conversions)

        assert "qu_factor_purchase_to_stock" not in products[0]

    def test_same_purchase_and_stock_unit_forces_one(self):
        api = make_api()
        products = [make_product(qu_purchase=QU_BOTTLE, qu_stock=QU_BOTTLE)]
        conversions = [make_conversion(QU_PACK, QU_BOTTLE, 18)]

        api.apply_quantity_unit_conversions(products, conversions)

        assert products[0]["qu_factor_purchase_to_stock"] == 1.0

    def test_cycle_does_not_hang(self):
        """A conversion loop with no path to the stock unit must terminate."""
        api = make_api()
        products = [make_product(qu_purchase=QU_PACK, qu_stock=99)]
        conversions = [
            make_conversion(QU_PACK, QU_BOX, 2),
            make_conversion(QU_BOX, QU_PACK, 0.5),
        ]

        api.apply_quantity_unit_conversions(products, conversions)

        assert "qu_factor_purchase_to_stock" not in products[0]

    def test_no_path_leaves_product_untouched(self):
        api = make_api()
        products = [make_product()]
        conversions = [make_conversion(QU_BOX, 42, 3)]

        api.apply_quantity_unit_conversions(products, conversions)

        assert "qu_factor_purchase_to_stock" not in products[0]

    def test_malformed_rows_are_skipped(self):
        """Garbage rows must not shadow the usable conversion."""
        api = make_api()
        products = [make_product()]
        conversions = [
            None,
            "not a dict",
            {"from_qu_id": "x", "to_qu_id": QU_BOTTLE, "factor": 2},
            make_conversion(QU_PACK, QU_BOTTLE, 0, product_id=1),
            make_conversion(QU_PACK, QU_BOTTLE, 18),
        ]

        api.apply_quantity_unit_conversions(products, conversions)

        assert products[0]["qu_factor_purchase_to_stock"] == 18.0

    def test_products_not_a_list_is_ignored(self):
        api = make_api()

        api.apply_quantity_unit_conversions(Exception("boom"), [])


# ── to_purchase_quantity ──────────────────────────────────────────────────────


class TestToPurchaseQuantity:
    @pytest.mark.parametrize(
        "amount,factor,expected",
        [
            (1, 1, 1),
            (3, 1, 3),
            ("3", 1, 3),
            (18, 18, 1),
            (5, 18, 1),
            (20, 18, 2),
            (36, 18, 2),
            (0, 18, 0),
            (0.5, 1, 1),
            ("0.5", 1, 1),
        ],
    )
    def test_rounds_up_to_whole_purchase_units(self, amount, factor, expected):
        api = make_api()
        assert api.to_purchase_quantity(amount, factor) == expected

    @pytest.mark.parametrize("factor", [0, None, "abc", -1])
    def test_unusable_factor_falls_back_to_one(self, factor):
        api = make_api()
        assert api.to_purchase_quantity(2, factor) == 2

    @pytest.mark.parametrize("amount", [None, "abc", "", -1])
    def test_unusable_amount_returns_zero(self, amount):
        api = make_api()
        assert api.to_purchase_quantity(amount, 18) == 0


# ── build_item_list — partial packs (issue #76 follow-up) ──────────────────


class TestBuildItemListQuantities:
    """A real purchase to stock factor must never display a quantity of zero."""

    def _make_data(self, amount, qty_factor):
        return {
            "shopping_lists": [{"id": 1, "name": "Liste principale"}],
            "products": [
                {
                    "id": "1",
                    "name": "Eau",
                    "qu_id_purchase": "3",
                    "qu_id_stock": "1",
                    "qu_factor_purchase_to_stock": qty_factor,
                },
            ],
            "shopping_list": [
                {
                    "id": "10",
                    "product_id": "1",
                    "shopping_list_id": 1,
                    "amount": amount,
                    "done": 0,
                },
            ],
        }

    def test_partial_pack_rounds_up_to_one(self):
        api = make_api()
        result = api.build_item_list(self._make_data(5, 18.0))
        assert result[0]["products"][0]["name"] == "Eau (x1)"

    def test_full_pack(self):
        api = make_api()
        result = api.build_item_list(self._make_data(18, 18.0))
        assert result[0]["products"][0]["name"] == "Eau (x1)"

    def test_more_than_one_pack_rounds_up(self):
        api = make_api()
        result = api.build_item_list(self._make_data(20, 18.0))
        assert result[0]["products"][0]["name"] == "Eau (x2)"

    def test_decimal_amount_does_not_crash(self):
        api = make_api()
        result = api.build_item_list(self._make_data("0.5", 1.0))
        assert result[0]["products"][0]["name"] == "Eau (x1)"
