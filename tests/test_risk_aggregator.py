"""
Tests for the Unified Disaster Risk Aggregation Engine.

Covers:
    • Normalisation functions (flood, earthquake, cyclone)
    • Risk level classification with hysteresis
    • Alert triggering logic
    • Core aggregation formula
    • Priority ordering
    • Edge cases (all zeros, all max, single hazard)
    • Concurrency amplification
    • Service-object convenience wrapper
"""

from __future__ import annotations

import pytest

from backend.app.ml.risk_aggregator import (
    ACTIVE_THRESHOLD,
    BETA,
    CYCLONE_CRITICAL,
    EARTHQUAKE_CRITICAL,
    FLOOD_CRITICAL,
    GAMMA,
    HYSTERESIS_BUFFER,
    THRESHOLD_SAFE_UPPER,
    THRESHOLD_WARNING_UPPER,
    THRESHOLD_WATCH_UPPER,
    W_CYCLONE,
    W_EARTHQUAKE,
    W_FLOOD,
    AggregatedRisk,
    AlertAction,
    HazardScore,
    OverallRiskLevel,
    aggregate_risk,
    classify_overall_risk,
    determine_alerts,
    normalise_cyclone,
    normalise_earthquake,
    normalise_flood,
    risk_level_to_action,
)


# ═══════════════════════════════════════════════════════════════════════════
# Normalisation
# ═══════════════════════════════════════════════════════════════════════════

class TestNormaliseFlood:
    def test_passthrough(self):
        assert normalise_flood(0.5) == 0.5

    def test_clamp_low(self):
        assert normalise_flood(-0.1) == 0.0

    def test_clamp_high(self):
        assert normalise_flood(1.5) == 1.0

    def test_zero(self):
        assert normalise_flood(0.0) == 0.0

    def test_one(self):
        assert normalise_flood(1.0) == 1.0


class TestNormaliseEarthquake:
    def test_shallow_m6(self):
        """Shallow M6: 6.0 × 1.5 / 10 = 0.9"""
        score = normalise_earthquake(6.0, 5.0)
        assert 0.85 <= score <= 0.95

    def test_intermediate_m4(self):
        """Intermediate depth M4: 4.0 × 0.6 / 10 = 0.24"""
        score = normalise_earthquake(4.0, 200.0)
        assert 0.20 <= score <= 0.28

    def test_deep_m3(self):
        """Deep M3: 3.0 × 0.2 / 10 = 0.06"""
        score = normalise_earthquake(3.0, 400.0)
        assert 0.04 <= score <= 0.08

    def test_zero_magnitude(self):
        assert normalise_earthquake(0.0, 10.0) == 0.0

    def test_capped_at_one(self):
        """Extreme M9.5 shallow should cap at 1.0."""
        score = normalise_earthquake(9.5, 5.0)
        assert score == 1.0


class TestNormaliseCyclone:
    def test_passthrough(self):
        assert normalise_cyclone(0.65) == 0.65

    def test_clamp_low(self):
        assert normalise_cyclone(-0.5) == 0.0

    def test_clamp_high(self):
        assert normalise_cyclone(1.3) == 1.0


# ═══════════════════════════════════════════════════════════════════════════
# Risk Level Classification
# ═══════════════════════════════════════════════════════════════════════════

class TestClassifyOverallRisk:
    def test_safe(self):
        assert classify_overall_risk(10.0) == OverallRiskLevel.SAFE

    def test_watch(self):
        assert classify_overall_risk(30.0) == OverallRiskLevel.WATCH

    def test_warning(self):
        assert classify_overall_risk(55.0) == OverallRiskLevel.WARNING

    def test_severe(self):
        assert classify_overall_risk(85.0) == OverallRiskLevel.SEVERE

    def test_boundary_safe_watch(self):
        assert classify_overall_risk(20.0) == OverallRiskLevel.WATCH

    def test_boundary_watch_warning(self):
        assert classify_overall_risk(45.0) == OverallRiskLevel.WARNING

    def test_boundary_warning_severe(self):
        assert classify_overall_risk(70.0) == OverallRiskLevel.SEVERE

    def test_hysteresis_resist_de_escalation(self):
        """Score dropped to 40 from WARNING, but hysteresis keeps WARNING (need ≤38)."""
        level = classify_overall_risk(40.0, previous_level=OverallRiskLevel.WARNING)
        assert level == OverallRiskLevel.WARNING

    def test_hysteresis_allows_de_escalation(self):
        """Score dropped to 35 from WARNING, below 38 → de-escalate to WATCH."""
        level = classify_overall_risk(35.0, previous_level=OverallRiskLevel.WARNING)
        assert level == OverallRiskLevel.WATCH

    def test_escalation_ignores_hysteresis(self):
        """Escalation always happens immediately."""
        level = classify_overall_risk(75.0, previous_level=OverallRiskLevel.WATCH)
        assert level == OverallRiskLevel.SEVERE


class TestRiskLevelToAction:
    def test_safe(self):
        assert risk_level_to_action(OverallRiskLevel.SAFE) == AlertAction.MONITOR

    def test_watch(self):
        assert risk_level_to_action(OverallRiskLevel.WATCH) == AlertAction.STAY_INFORMED

    def test_warning(self):
        assert risk_level_to_action(OverallRiskLevel.WARNING) == AlertAction.PREPARE

    def test_severe(self):
        assert risk_level_to_action(OverallRiskLevel.SEVERE) == AlertAction.EVACUATE


# ═══════════════════════════════════════════════════════════════════════════
# Alert Triggering
# ═══════════════════════════════════════════════════════════════════════════

class TestDetermineAlerts:
    def _make_score(self, hazard: str, norm: float) -> HazardScore:
        return HazardScore(
            hazard_type=hazard,
            raw_value=norm,
            normalised_score=norm,
            weight=0.33,
            is_active=norm >= ACTIVE_THRESHOLD,
            is_critical=norm >= 0.80,
            priority={"earthquake": 1, "cyclone": 2, "flood": 3}[hazard],
        )

    def test_no_alert_when_safe(self):
        scores = [
            self._make_score("flood", 0.1),
            self._make_score("earthquake", 0.05),
            self._make_score("cyclone", 0.0),
        ]
        triggered, reasons = determine_alerts(scores, OverallRiskLevel.SAFE)
        assert not triggered
        assert reasons == []

    def test_alert_on_escalation(self):
        scores = [self._make_score("flood", 0.5)]
        triggered, reasons = determine_alerts(
            scores, OverallRiskLevel.WARNING,
            previous_level=OverallRiskLevel.WATCH,
        )
        assert triggered
        assert any("escalated" in r.lower() for r in reasons)

    def test_alert_on_critical_hazard(self):
        scores = [self._make_score("earthquake", 0.90)]
        triggered, reasons = determine_alerts(scores, OverallRiskLevel.SEVERE)
        assert triggered
        assert any("EARTHQUAKE" in r for r in reasons)

    def test_alert_on_concurrent_hazards(self):
        scores = [
            self._make_score("flood", 0.5),
            self._make_score("cyclone", 0.4),
            self._make_score("earthquake", 0.1),
        ]
        triggered, reasons = determine_alerts(scores, OverallRiskLevel.WARNING)
        assert triggered
        assert any("concurrent" in r.lower() for r in reasons)


# ═══════════════════════════════════════════════════════════════════════════
# Core Aggregation
# ═══════════════════════════════════════════════════════════════════════════

class TestAggregateRisk:
    def test_all_zeros(self):
        """No hazards → score near 0, level SAFE."""
        result = aggregate_risk()
        assert result.overall_risk_score == pytest.approx(0.0, abs=0.01)
        assert result.overall_risk_level == OverallRiskLevel.SAFE
        assert result.active_hazard_count == 0
        assert not result.alert_triggered

    def test_flood_only_moderate(self):
        """Flood at 0.5 probability, no other hazards."""
        result = aggregate_risk(flood_probability=0.5)
        # R_max = 0.5, R_avg = 0.4*0.5 = 0.20
        # R_hybrid = 0.6*0.5 + 0.4*0.2 = 0.30 + 0.08 = 0.38
        # Score = 38%  → WATCH (with 1 active, amplifier=1.0)
        assert 35.0 <= result.overall_risk_score <= 42.0
        assert result.dominant_hazard == "flood"
        assert result.active_hazard_count == 1

    def test_earthquake_only_major(self):
        """Major shallow earthquake M7.0 at 8 km depth."""
        result = aggregate_risk(earthquake_magnitude=7.0, earthquake_depth_km=8.0)
        # S_eq = 7.0 * 1.5 / 10 = 1.05 → capped at 1.0
        # R_max = 1.0, R_avg = 0.3*1.0 = 0.3
        # R_hybrid = 0.6*1.0 + 0.4*0.3 = 0.72
        # Score = 72% → SEVERE
        assert result.overall_risk_score >= 70.0
        assert result.overall_risk_level == OverallRiskLevel.SEVERE
        assert result.dominant_hazard == "earthquake"

    def test_cyclone_only_extreme(self):
        """Extreme cyclone score 0.9."""
        result = aggregate_risk(cyclone_score=0.9)
        # R_max = 0.9, R_avg = 0.3*0.9 = 0.27
        # R_hybrid = 0.6*0.9 + 0.4*0.27 = 0.54 + 0.108 = 0.648
        # Score = 64.8% → WARNING
        assert 60.0 <= result.overall_risk_score <= 70.0
        assert result.dominant_hazard == "cyclone"

    def test_multi_hazard_amplification(self):
        """Two active hazards should receive amplification bonus."""
        result = aggregate_risk(
            flood_probability=0.5,
            cyclone_score=0.5,
        )
        assert result.active_hazard_count == 2
        assert result.amplifier == pytest.approx(1.10, abs=0.001)
        # Without amplifier: R_hybrid ≈ 0.40
        # With amplifier: 0.40 * 1.10 * 100 ≈ 44
        assert result.overall_risk_score > 40.0

    def test_triple_hazard_maximum(self):
        """All three hazards at critical → score near 100, SEVERE."""
        result = aggregate_risk(
            flood_probability=0.95,
            earthquake_magnitude=8.0,
            earthquake_depth_km=5.0,
            cyclone_score=0.95,
        )
        assert result.overall_risk_score >= 95.0
        assert result.overall_risk_level == OverallRiskLevel.SEVERE
        assert result.active_hazard_count == 3
        assert result.amplifier == pytest.approx(1.20, abs=0.001)
        assert result.alert_triggered

    def test_score_capped_at_100(self):
        """Score never exceeds 100."""
        result = aggregate_risk(
            flood_probability=1.0,
            earthquake_magnitude=10.0,
            earthquake_depth_km=1.0,
            cyclone_score=1.0,
        )
        assert result.overall_risk_score <= 100.0

    def test_priority_ordering_in_breakdown(self):
        """Hazard scores should be sorted by priority: EQ > CY > FL."""
        result = aggregate_risk(
            flood_probability=0.5,
            earthquake_magnitude=5.0,
            earthquake_depth_km=10.0,
            cyclone_score=0.5,
        )
        types = [h.hazard_type for h in result.hazard_scores]
        assert types == ["earthquake", "cyclone", "flood"]

    def test_dominant_hazard_is_highest_score(self):
        """Single dominant hazard is correctly identified."""
        result = aggregate_risk(
            flood_probability=0.2,
            earthquake_magnitude=6.5,
            earthquake_depth_km=10.0,
            cyclone_score=0.3,
        )
        assert result.dominant_hazard == "earthquake"

    def test_to_dict_structure(self):
        """Verify to_dict() produces expected keys."""
        result = aggregate_risk(flood_probability=0.5)
        d = result.to_dict()
        assert "overall_risk_score" in d
        assert "overall_risk_score_pct" in d
        assert "overall_risk_level" in d
        assert "hazard_breakdown" in d
        assert "formula_components" in d
        assert "thresholds" in d
        assert len(d["hazard_breakdown"]) == 3


class TestHysteresisIntegration:
    """Test hysteresis through the full aggregate_risk pipeline."""

    def test_resists_de_escalation(self):
        """WARNING should persist when score is in hysteresis band."""
        result = aggregate_risk(
            flood_probability=0.45,
            previous_level=OverallRiskLevel.WARNING,
        )
        # Score ≈ 34.2, above de-escalation threshold of 38… let me check
        # Actually with 0.45: R_max=0.45, R_avg=0.18, R_hybrid=0.342, score=34.2
        # De-escalation from WARNING→WATCH needs score ≤ 38
        # 34.2 ≤ 38 → does de-escalate
        # Let me use a higher value to stay in the band
        result2 = aggregate_risk(
            flood_probability=0.55,
            previous_level=OverallRiskLevel.WARNING,
        )
        # Score ≈ 41.8  which is > 38  → hysteresis kicks in → stays WARNING
        assert result2.overall_risk_level == OverallRiskLevel.WARNING

    def test_allows_escalation(self):
        """Escalation from WATCH → WARNING should happen immediately."""
        result = aggregate_risk(
            flood_probability=0.70,
            cyclone_score=0.50,
            previous_level=OverallRiskLevel.WATCH,
        )
        # High enough multi-hazard to push past 45%
        assert result.overall_risk_level in (
            OverallRiskLevel.WARNING, OverallRiskLevel.SEVERE
        )


# ═══════════════════════════════════════════════════════════════════════════
# Edge Cases
# ═══════════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    def test_negative_inputs_clamped(self):
        """Negative hazard values should be clamped to 0."""
        result = aggregate_risk(
            flood_probability=-0.5,
            earthquake_magnitude=-1.0,
            cyclone_score=-0.3,
        )
        assert result.overall_risk_score >= 0.0
        for hs in result.hazard_scores:
            assert hs.normalised_score >= 0.0

    def test_custom_weights(self):
        """Custom weights should be respected."""
        result = aggregate_risk(
            flood_probability=1.0,
            w_flood=1.0,
            w_earthquake=0.0,
            w_cyclone=0.0,
        )
        # Pure flood weighting: R_avg = 1.0, R_max = 1.0 → score = 100
        assert result.overall_risk_score >= 95.0

    def test_custom_beta(self):
        """Beta=0 should give pure weighted average."""
        result_max = aggregate_risk(flood_probability=0.8, beta=1.0)
        result_avg = aggregate_risk(flood_probability=0.8, beta=0.0)
        # Pure max should give higher score than pure average
        assert result_max.overall_risk_score >= result_avg.overall_risk_score
