# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Focused tests for the paper/archive/corrected reconciliation ledger."""

from __future__ import annotations

import json
import hashlib
import os
import tempfile
import unittest
from typing import Any
from unittest.mock import patch

import pandas as pd

from benchmark_scripts import reconcile_results


class ReconcileResultsTest(unittest.TestCase):
    """Validate historical recomputation and strict estimand matching."""

    def test_audited_claim_digest_excludes_mutable_audit_outcomes(self) -> None:
        published: dict[str, Any] = reconcile_results._base_row(
            source_state="published",
            source_locator="audited_claim_set/table_2",
            source_sha256="",
            paper_location="Table 2",
            benchmark="boolq",
            pregrouper="sentence",
            pair_set="all",
            metric="F_pred",
            aggregation="row_pooled_then_model_pair_distribution",
            value_component="median",
            value=0.709,
            display_value=".709",
            n_pairs=55,
            effective_pairs=(),
            data_status="published",
        )
        claim: dict[str, Any] = reconcile_results._claim_row(
            claim_id="paper.fixture",
            claim_group="fixture",
            source_location="paper/main_text",
            source_state="archive",
            benchmark="boolq",
            pregrouper="sentence",
            segment_grid_id="grid",
            cohort="pair",
            scope="not_applicable",
            contrast="true_minus_false",
            representation="scalar_logodds",
            metric="F_pred",
            statistic="pearson_r2",
            aggregation="prompt_level",
            missingness_policy="pairwise_drop_nonfinite",
            pair_set="single_pair",
            model_s="a",
            model_t="b",
            value_component="point",
            expected_display=".700",
            expected_value=0.7,
            tolerance=0.0005,
            actual_value=0.7,
            input_locators=["archive/input.tsv"],
        )
        baseline: str = reconcile_results.audited_claim_set_sha256([published], [claim])
        changed_outcome: dict[str, Any] = dict(claim)
        changed_outcome["actual_value"] = 0.1
        changed_outcome["passed"] = False
        self.assertEqual(
            baseline,
            reconcile_results.audited_claim_set_sha256([published], [changed_outcome]),
        )
        changed_claim: dict[str, Any] = dict(claim)
        changed_claim["expected_display"] = ".701"
        self.assertNotEqual(
            baseline,
            reconcile_results.audited_claim_set_sha256([published], [changed_claim]),
        )

    def _race_grid_fixture(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        prompts: tuple[dict[str, Any], ...] = (
            {
                "prompt_idx": 0,
                "article": "Article.",
                "question": "Question",
                "A": "One",
                "B": "Two",
                "C": "Three",
                "D": "Four",
                "historical_count": 9,
                "corrected_count": 13,
            },
            {
                "prompt_idx": 1,
                "article": "Article.",
                "question": "Question?",
                "A": "One.",
                "B": "Two.",
                "C": "Three.",
                "D": "Four.",
                "historical_count": 13,
                "corrected_count": 13,
            },
        )
        historical_rows: list[dict[str, object]] = []
        corrected_rows: list[dict[str, object]] = []
        for prompt in prompts:
            prompt_idx: int = int(prompt["prompt_idx"])
            historical_count: int = int(prompt["historical_count"])
            corrected_count: int = int(prompt["corrected_count"])
            historical_rows.extend(
                {
                    "prompt_idx": prompt_idx,
                    "segment_idx": segment_idx,
                    "n_segments": historical_count,
                    **{
                        field: prompt[field]
                        for field in ("article", "question", "A", "B", "C", "D")
                    },
                }
                for segment_idx in range(historical_count)
            )
            corrected_rows.extend(
                {
                    "prompt_idx": prompt_idx,
                    "seg_idx": segment_idx,
                    "n_segments": corrected_count,
                    "message_role": "system" if segment_idx < 3 else "user",
                }
                for segment_idx in range(corrected_count)
            )
        return pd.DataFrame(historical_rows), pd.DataFrame(corrected_rows)

    def test_gold_race_grid_structure_rejects_each_identity_failure(self) -> None:
        historical, corrected = self._race_grid_fixture()
        reconcile_results._validate_race_grid_structure(historical, corrected, {0, 1})

        bad_prompt_ids: pd.DataFrame = corrected.copy()
        bad_prompt_ids.loc[bad_prompt_ids["prompt_idx"].eq(1), "prompt_idx"] = 2
        duplicate_key: pd.DataFrame = corrected.copy()
        duplicate_key.loc[duplicate_key.index[-1], "seg_idx"] = 0
        missing_historical_key: pd.DataFrame = historical.copy()
        missing_historical_key.loc[
            missing_historical_key["prompt_idx"].eq(0)
            & missing_historical_key["segment_idx"].eq(8),
            "segment_idx",
        ] = 99
        wrong_template_count: pd.DataFrame = historical.drop(
            historical[
                historical["prompt_idx"].eq(0) & historical["segment_idx"].eq(8)
            ].index
        ).copy()
        wrong_template_count.loc[
            wrong_template_count["prompt_idx"].eq(0), "n_segments"
        ] = 8
        wrong_system_scope: pd.DataFrame = corrected.copy()
        wrong_system_scope.loc[0, "message_role"] = "user"

        failures: tuple[tuple[pd.DataFrame, pd.DataFrame, str], ...] = (
            (historical, bad_prompt_ids, "prompt IDs"),
            (historical, duplicate_key, "not unique"),
            (missing_historical_key, corrected, "not retained"),
            (wrong_template_count, corrected, "template was not reconstructed"),
            (historical, wrong_system_scope, "system-message grid"),
        )
        for historical_input, corrected_input, error in failures:
            with self.subTest(error=error), self.assertRaisesRegex(ValueError, error):
                reconcile_results._validate_race_grid_structure(
                    historical_input, corrected_input, {0, 1}
                )

    def _cohort(self) -> reconcile_results.LegacyCohort:
        mapping: dict[str, str] = {
            f"open_{index}": model for index, model in enumerate(reconcile_results.OPEN)
        }
        mapping.update(
            {
                f"hosted_{index}": model
                for index, model in enumerate(reconcile_results.HOSTED)
            }
        )
        return reconcile_results.LegacyCohort(
            mapping, reconcile_results.OPEN, reconcile_results.HOSTED
        )

    def test_race_w_norm_claims_bind_every_open_model(self) -> None:
        cohort: reconcile_results.LegacyCohort = self._cohort()
        with tempfile.TemporaryDirectory() as temporary:
            archive_dir: str = os.path.join(temporary, "archive")
            corrected_dir: str = os.path.join(temporary, "corrected")
            race_dir: str = os.path.join(corrected_dir, "race", "sentence")
            os.makedirs(archive_dir)
            os.makedirs(race_dir)
            prompt_ids: list[int] = list(range(4_934))
            archive_data: dict[str, list[float] | list[int]] = {
                "prompt_idx": prompt_ids
            }
            for model_index, model in enumerate(cohort.open_models):
                values: list[float] = [
                    float(model_index + 1 + prompt_idx % 3) for prompt_idx in prompt_ids
                ]
                archive_data[f"{cohort.archive_prefix(model)}_w_norm"] = values
                pd.DataFrame({"prompt_idx": prompt_ids, "w_norm": values}).to_csv(
                    os.path.join(race_dir, f"{model}_segment.tsv.gz"),
                    sep="\t",
                    index=False,
                )
            pd.DataFrame(archive_data).to_csv(
                os.path.join(archive_dir, "race_sentence_consolidated.tsv"),
                sep="\t",
                index=False,
            )

            claims, inputs = reconcile_results._race_w_norm_claims(
                archive_dir,
                corrected_dir,
                cohort,
            )
            self.assertEqual(len(cohort.open_models), len(claims))
            self.assertTrue(all(row["actual_value"] == 0.0 for row in claims))
            self.assertEqual(len(cohort.open_models), len(inputs))

            first_model: str = cohort.open_models[0]
            altered_path: str = os.path.join(race_dir, f"{first_model}_segment.tsv.gz")
            altered: pd.DataFrame = pd.read_csv(altered_path, sep="\t")
            altered.loc[0, "w_norm"] += 0.01
            altered.to_csv(altered_path, sep="\t", index=False)
            altered_claims, _ = reconcile_results._race_w_norm_claims(
                archive_dir,
                corrected_dir,
                cohort,
            )
            altered_claim: dict[str, Any] = next(
                row for row in altered_claims if row["model_s"] == first_model
            )
            self.assertAlmostEqual(0.01, altered_claim["actual_value"])
            self.assertFalse(altered_claim["passed"])

    def _wide_frame(self) -> pd.DataFrame:
        cohort: reconcile_results.LegacyCohort = self._cohort()
        rows: list[dict[str, float | int]] = []
        for prompt_idx in range(4):
            for segment_idx in range(4):
                row: dict[str, float | int] = {
                    "prompt_idx": prompt_idx,
                    "segment_idx": segment_idx,
                    "n_segments": 4,
                }
                for model_idx, model in enumerate(cohort.all_models):
                    prefix: str = cohort.archive_prefix(model)
                    scale: float = float(model_idx + 1)
                    row[f"{prefix}_orig_logodds"] = scale * (prompt_idx + 1)
                    row[f"{prefix}_ablation"] = scale * (segment_idx + 1)
                    if model in cohort.open_models:
                        row[f"{prefix}_w_norm"] = 1.0
                        row[f"{prefix}_dn_postnorm"] = scale * (segment_idx + 1)
                        row[f"{prefix}_wdz_postnorm"] = scale * (segment_idx + 1) ** 2
                        row[f"{prefix}_attn_rollout"] = scale * (segment_idx + 1)
                        row[f"{prefix}_attn_mean"] = scale * (segment_idx + 1)
                        row[f"{prefix}_attn_max"] = scale * (segment_idx + 1)
                rows.append(row)
        return pd.DataFrame(rows)

    def _write_race_raw_files(self, archive_dir: str) -> None:
        """Write a tiny complete four-label fixture under neutral public names."""
        raw_dir: str = os.path.join(archive_dir, "race_raw")
        os.makedirs(raw_dir)
        for model_index, model in enumerate(reconcile_results.ALL):
            payload: list[dict[str, Any]] = []
            for prompt_idx in range(4):
                baseline: dict[str, float] = {
                    label: float((model_index + 1) * (prompt_idx + label_index + 1))
                    for label_index, label in enumerate(("A", "B", "C", "D"))
                }
                payload.append(
                    {
                        "prompt_idx": prompt_idx,
                        "answer": "A",
                        "orig_label_logprobs": baseline,
                        "n_segments": 4,
                        "ablated_label_logprobs": [
                            {
                                label: value - 0.1 * (segment_idx + label_index + 1)
                                for label_index, (label, value) in enumerate(
                                    baseline.items()
                                )
                            }
                            for segment_idx in range(4)
                        ],
                    }
                )
            with open(
                os.path.join(raw_dir, f"{model}.json"), "w", encoding="utf-8"
            ) as output:
                json.dump(payload, output)

    def test_archive_recomputation_uses_historical_aggregation(self) -> None:
        values: dict[str, dict[str, float]] = reconcile_results.recompute_archive(
            self._wide_frame(), self._cohort()
        )
        self.assertEqual(55, len(values["F_pred:all"]))
        self.assertEqual(55, len(values["F_attr:all"]))
        self.assertEqual(10, len(values["F_attn_mean:all"]))
        self.assertEqual(50, len(values["F_mag_to_attr:all"]))
        for key, estimates in values.items():
            if (
                key.split(":", 1)[0] in reconcile_results.REPRESENTATION_METRICS
                and not estimates
            ):
                continue
            self.assertTrue(estimates, key)
            self.assertTrue(
                all(abs(value - 1.0) < 1e-12 for value in estimates.values()), key
            )
        prompt_equal: dict[str, dict[str, float]] = reconcile_results.recompute_archive(
            self._wide_frame(),
            self._cohort(),
            tuple(reconcile_results.PROMPT_EQUAL_EXPECTATIONS),
            transfer_aggregation="prompt_equal_mean_r2",
        )
        self.assertTrue(
            all(
                abs(value - 1.0) < 1e-12
                for estimates in prompt_equal.values()
                for value in estimates.values()
            )
        )

    def test_historical_race_loader_preserves_missing_coordinate_keys(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path: str = os.path.join(temporary, "race.json")
            payload: list[dict[str, Any]] = []
            for prompt_idx in range(3):
                original: dict[str, float] = {
                    "A": -1.0 - prompt_idx,
                    "B": -2.0,
                    "C": -3.0,
                    "D": -4.0,
                }
                ablated: list[dict[str, float | None]] = [
                    dict(original),
                    dict(original),
                    dict(original),
                ]
                if prompt_idx == 0:
                    ablated[0]["D"] = None
                payload.append(
                    {
                        "prompt_idx": prompt_idx,
                        "orig_label_logprobs": original,
                        "n_segments": 3,
                        "ablated_label_logprobs": ablated,
                    }
                )
            with open(path, "w", encoding="utf-8") as output:
                json.dump(payload, output)
            signals = reconcile_results._historical_race_signal(path)
            self.assertEqual(3, len(signals.raw_prediction_keys))
            self.assertEqual(9, len(signals.raw_attribution_keys))
            self.assertEqual(8, len(signals.attribution))

    def test_historical_finite_extreme_matches_public_sensitivity(self) -> None:
        signal: pd.Series = pd.Series(
            [-float("inf"), -2.0, 3.0, float("inf"), float("nan")]
        )
        observed: pd.Series = reconcile_results._finite_extreme_signal(signal)
        self.assertAlmostEqual(-2.03, observed.iloc[0])
        self.assertAlmostEqual(3.04, observed.iloc[3])
        self.assertTrue(pd.isna(observed.iloc[4]))

    def test_expected_discrepancy_is_an_executable_claim(self) -> None:
        claim: dict[str, Any] = reconcile_results._claim_row(
            claim_id="fixture.discrepancy",
            claim_group="fixture",
            source_location="fixture/source",
            source_state="archive",
            benchmark="boolq",
            pregrouper="sentence",
            segment_grid_id="fixture",
            cohort="single_pair",
            scope="not_applicable",
            contrast="canonical",
            representation="scalar",
            metric="F_pred",
            statistic="pearson_r2",
            aggregation="prompt_level_single_pair",
            missingness_policy="pairwise_drop_nonfinite",
            pair_set="single_pair",
            model_s="a",
            model_t="b",
            value_component="point",
            expected_display=".684",
            expected_value=0.684,
            tolerance=0.0005,
            actual_value=0.679,
            input_locators=["archive/input.tsv"],
            expected_relation="outside_tolerance",
        )
        self.assertTrue(claim["passed"])

    def test_comparable_delta_rejects_different_estimands(self) -> None:
        left: dict[str, Any] = {"estimand_id": "a", "value": 0.4}
        right: dict[str, Any] = {"estimand_id": "b", "value": 0.2}
        with self.assertRaisesRegex(ValueError, "nonmatching estimands"):
            reconcile_results.comparable_delta(left, right)
        right["estimand_id"] = "a"
        self.assertAlmostEqual(0.2, reconcile_results.comparable_delta(left, right))

    def test_corrected_schema_requires_scope_and_contrast(self) -> None:
        frame: pd.DataFrame = pd.DataFrame(
            columns=[
                "benchmark",
                "model_s",
                "model_t",
                "metric",
                "statistic",
                "f_point",
            ]
        )
        with self.assertRaisesRegex(ValueError, "estimand metadata"):
            reconcile_results._validate_corrected_schema(frame, "corrected/f_table.tsv")

    def test_scope_and_contrast_record_declared_and_executed_estimands(self) -> None:
        self.assertEqual(
            (
                "not_applicable",
                "not_applicable",
                "entailment_contradiction",
                "entailment_neutral",
            ),
            reconcile_results._scope_and_contrast("anli_r1", "F_pred"),
        )
        self.assertEqual(
            ("user", "all", "entailment_contradiction", "entailment_neutral"),
            reconcile_results._scope_and_contrast("anli_r1", "F_align"),
        )
        self.assertEqual(
            ("user", "all", "not_applicable", "not_applicable"),
            reconcile_results._scope_and_contrast("anli_r1", "F_attn_mean"),
        )

    def test_boolq_word_historical_attention_has_distinct_estimand(self) -> None:
        self.assertEqual(
            "boolq_word_historical_sentence_contaminated_attention",
            reconcile_results._grid_id("boolq", "word", "archive", "F_attn_mean"),
        )
        self.assertEqual(
            "boolq_word_paper_full_dialog",
            reconcile_results._grid_id("boolq", "word", "corrected", "F_attn_mean"),
        )
        self.assertEqual(
            "boolq_word_paper_full_dialog",
            reconcile_results._grid_id("boolq", "word", "archive", "F_attr"),
        )
        contaminated: list[dict[str, Any]] = [
            row
            for row in reconcile_results._published_rows()
            if row["benchmark"] == "boolq"
            and row["pregrouper"] == "word"
            and row["metric"]
            in reconcile_results.BOOLQ_WORD_CONTAMINATED_ATTENTION_METRICS
        ]
        self.assertEqual(6, len(contaminated))
        self.assertTrue(
            all(
                row["data_status"]
                == "published_invalid_sentence_attention_contamination"
                for row in contaminated
            )
        )
        self.assertTrue(all("sentence-level" in row["note"] for row in contaminated))

    def test_boolq_word_contamination_claims_require_shared_sentence_mask(
        self,
    ) -> None:
        cohort: reconcile_results.LegacyCohort = self._cohort()
        coordinate_columns: list[str] = ["prompt_idx", "segment_idx"]
        word: pd.DataFrame = pd.DataFrame(
            [(0, 0), (0, 2), (1, 1), (2, 5)], columns=coordinate_columns
        )
        sentence: pd.DataFrame = pd.DataFrame(
            [(0, 0), (0, 1), (1, 1), (2, 5)], columns=coordinate_columns
        )
        qwen_models: list[str] = [
            model for model in cohort.open_models if model.startswith("qwen2.5-")
        ]
        for model in qwen_models:
            prefix: str = cohort.archive_prefix(model)
            for variant in ("rollout", "mean", "max"):
                column: str = f"{prefix}_attn_{variant}"
                word[column] = [1.0, float("nan"), 2.0, 3.0]
                sentence[column] = [1.0, 10.0, 2.0, 3.0]
        with tempfile.TemporaryDirectory() as temporary:
            word.to_csv(
                os.path.join(temporary, "boolq_word_consolidated.tsv"),
                sep="\t",
                index=False,
            )
            sentence.to_csv(
                os.path.join(temporary, "boolq_sentence_consolidated.tsv"),
                sep="\t",
                index=False,
            )
            claims: list[dict[str, Any]] = (
                reconcile_results._boolq_word_attention_contamination_claims(
                    temporary, cohort
                )
            )
            by_component: dict[str, dict[str, Any]] = {
                str(claim["value_component"]): claim for claim in claims
            }
            self.assertEqual(3.0, by_component["coordinates"]["actual_value"])
            self.assertEqual(3, by_component["coordinates"]["n_prompts"])
            self.assertAlmostEqual(1.0, by_component["minimum"]["actual_value"])
            self.assertEqual(12, by_component["minimum"]["n_pairs"])
            self.assertEqual(36, by_component["minimum"]["n_observations"])

            first_column: str = f"{cohort.archive_prefix(qwen_models[0])}_attn_mean"
            word.loc[word["segment_idx"].eq(2), first_column] = 4.0
            word.to_csv(
                os.path.join(temporary, "boolq_word_consolidated.tsv"),
                sep="\t",
                index=False,
            )
            mismatched: list[dict[str, Any]] = (
                reconcile_results._boolq_word_attention_contamination_claims(
                    temporary, cohort
                )
            )
            self.assertEqual("", mismatched[0]["actual_value"])
            self.assertFalse(mismatched[0]["passed"])

    def test_race_and_transfer_publication_estimands_are_exact(self) -> None:
        rows: list[dict[str, Any]] = reconcile_results._published_rows()
        race_metrics: set[str] = {
            str(row["metric"])
            for row in rows
            if row["paper_location"] == "Table 2" and row["benchmark"] == "race"
        }
        self.assertEqual({"F_pred", "F_attr"}, race_metrics)
        published_transfer: dict[str, Any] = next(
            row
            for row in rows
            if row["paper_location"] == "Table 1" and row["metric"] == "F_align_to_attr"
        )
        self.assertEqual(
            "row_pooled_then_model_pair_distribution",
            published_transfer["aggregation"],
        )
        self.assertNotEqual(
            published_transfer["aggregation"],
            reconcile_results._corrected_aggregation("prompt_equal_mean_r2"),
        )

    def test_reduced_corrected_pair_set_is_marked_incomplete(self) -> None:
        frame: pd.DataFrame = pd.DataFrame(
            [
                {
                    "benchmark": "boolq",
                    "pregrouper": "sentence",
                    "scope": "all",
                    "contrast": "canonical",
                    "api_infinity_policy": "drop",
                    "aggregation": "row_pooled",
                    "model_s": reconcile_results.OPEN[0],
                    "model_t": reconcile_results.OPEN[1],
                    "metric": "F_pred",
                    "statistic": "pearson_r2",
                    "f_point": 0.5,
                }
            ]
        )
        row: dict[str, Any] | None = reconcile_results._summarize_corrected(
            frame,
            "corrected/f_table.tsv",
            "digest",
            reconcile_results.BENCHMARKS[0],
            "Table 2",
            "all",
            "F_pred",
            "median",
        )
        self.assertIsNotNone(row)
        assert row is not None
        self.assertEqual("incomplete_pair_set", row["data_status"])
        self.assertEqual(1, row["n_pairs"])

    def test_corrected_aggregation_must_be_single_and_known(self) -> None:
        base: dict[str, Any] = {
            "benchmark": "boolq",
            "pregrouper": "sentence",
            "scope": "all",
            "contrast": "canonical",
            "api_infinity_policy": "drop",
            "aggregation": "unknown",
            "model_s": reconcile_results.OPEN[0],
            "model_t": reconcile_results.OPEN[1],
            "metric": "F_pred",
            "statistic": "pearson_r2",
            "f_point": 0.5,
        }
        with self.assertRaisesRegex(ValueError, "Unknown corrected aggregation"):
            reconcile_results._summarize_corrected(
                pd.DataFrame([base]),
                "corrected/f_table.tsv",
                "digest",
                reconcile_results.BENCHMARKS[0],
                "Table 2",
                "all",
                "F_pred",
                "median",
            )
        mixed: dict[str, Any] = dict(base)
        base["aggregation"] = "row_pooled"
        mixed["aggregation"] = "prompt_equal_mean_r2"
        mixed["model_t"] = reconcile_results.OPEN[2]
        with self.assertRaisesRegex(ValueError, "missing or mixed aggregation"):
            reconcile_results._summarize_corrected(
                pd.DataFrame([base, mixed]),
                "corrected/f_table.tsv",
                "digest",
                reconcile_results.BENCHMARKS[0],
                "Table 2",
                "all",
                "F_pred",
                "median",
            )

    def test_stale_corrected_sidecar_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            table_path: str = os.path.join(temporary, "f_table.tsv")
            with open(table_path, "w", encoding="utf-8") as output:
                output.write("value\n1\n")
            with open(f"{table_path}.provenance.json", "w", encoding="utf-8") as output:
                json.dump(
                    {
                        "output": {"sha256": "stale"},
                        "inputs": {"raw": {"path": "missing.tsv", "sha256": "stale"}},
                    },
                    output,
                )
            valid, checks, _ = reconcile_results._verify_derived_sidecar(
                temporary, "corrected/f_table.tsv", table_path
            )
            self.assertFalse(valid)
            self.assertTrue(any(not check["passed"] for check in checks))

    def test_cli_exit_status_reflects_required_checks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            table_path: str = os.path.join(temporary, "reconciliation.tsv")
            claims_path: str = os.path.join(temporary, "historical_claims.tsv")
            manifest_path: str = os.path.join(temporary, "reconciliation_manifest.json")
            checks_path: str = os.path.join(temporary, "reconciliation_checks.json")
            for path in (table_path, claims_path, manifest_path):
                with open(path, "w", encoding="utf-8") as output:
                    output.write("\n")
            arguments: list[str] = [
                "reconcile_results",
                "--archive-dir",
                "archive",
                "--corrected-results-dir",
                "corrected",
                "--output-dir",
                "output",
            ]
            paths: tuple[str, str, str, str] = (
                table_path,
                claims_path,
                manifest_path,
                checks_path,
            )
            with open(checks_path, "w", encoding="utf-8") as output:
                json.dump({"all_passed": True}, output)
            with (
                patch("sys.argv", arguments),
                patch.object(reconcile_results, "reconcile", return_value=paths),
            ):
                self.assertIsNone(reconcile_results.main())

            with open(checks_path, "w", encoding="utf-8") as output:
                json.dump({"all_passed": False}, output)
            with (
                patch("sys.argv", arguments),
                patch.object(reconcile_results, "reconcile", return_value=paths),
                self.assertRaises(SystemExit) as raised,
            ):
                reconcile_results.main()
            self.assertEqual(1, raised.exception.code)

    def test_end_to_end_uses_portable_locators_and_guards_race(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            archive_dir: str = os.path.join(temporary, "legacy")
            corrected_dir: str = os.path.join(temporary, "corrected")
            output_dir: str = os.path.join(temporary, "output")
            os.makedirs(archive_dir)
            os.makedirs(corrected_dir)
            cohort: reconcile_results.LegacyCohort = self._cohort()
            inverse_mapping: dict[str, str] = cohort.archive_to_public
            with open(
                os.path.join(archive_dir, "reconciliation_models.json"),
                "w",
                encoding="utf-8",
            ) as output:
                json.dump(
                    {
                        "archive_to_public_model": inverse_mapping,
                        "open_models": list(cohort.open_models),
                        "hosted_models": list(cohort.hosted_models),
                    },
                    output,
                )
            wide: pd.DataFrame = self._wide_frame()
            for spec in reconcile_results.BENCHMARKS:
                wide.to_csv(
                    os.path.join(archive_dir, spec.filename), sep="\t", index=False
                )
            self._write_race_raw_files(archive_dir)
            race_manifest_dir: str = os.path.join(corrected_dir, "race", "sentence")
            os.makedirs(race_manifest_dir)
            pd.DataFrame(
                [
                    {
                        "prompt_idx": prompt_idx,
                        "seg_idx": segment_idx,
                        "n_segments": 4,
                    }
                    for prompt_idx in range(4)
                    for segment_idx in range(4)
                ]
            ).to_csv(
                os.path.join(race_manifest_dir, "segments.tsv.gz"),
                sep="\t",
                index=False,
            )

            columns: list[str] = [
                "artifact_role",
                "candidate_id",
                "cohort",
                "benchmark",
                "pregrouper",
                "scope",
                "requested_scope",
                "resolved_scope",
                "contrast",
                "requested_contrast",
                "resolved_source_contrast",
                "resolved_target_contrast",
                "readout_contrast",
                "api_infinity_policy",
                "aggregation",
                "availability_status",
                "unavailable_reason",
                "model_s",
                "model_t",
                "metric",
                "statistic",
                "f_point",
            ]
            pd.DataFrame(
                [
                    {
                        "benchmark": "boolq",
                        "pregrouper": "sentence",
                        "scope": "all",
                        "contrast": "canonical",
                        "api_infinity_policy": "drop",
                        "aggregation": "row_pooled",
                        "model_s": model_s,
                        "model_t": model_t,
                        "metric": "F_pred",
                        "statistic": "pearson_r2",
                        "f_point": 0.7,
                    }
                    for index, model_s in enumerate(reconcile_results.ALL)
                    for model_t in reconcile_results.ALL[index + 1 :]
                ],
                columns=columns,
            ).to_csv(os.path.join(corrected_dir, "f_table.tsv"), sep="\t", index=False)
            pd.DataFrame(
                [
                    {
                        "benchmark": "race",
                        "pregrouper": "sentence",
                        "scope": "all",
                        "contrast": "canonical",
                        "api_infinity_policy": "drop",
                        "aggregation": "row_pooled",
                        "model_s": model_s,
                        "model_t": model_t,
                        "metric": "F_pred",
                        "statistic": "pearson_r2",
                        "f_point": 0.75,
                    }
                    for index, model_s in enumerate(reconcile_results.ALL)
                    for model_t in reconcile_results.ALL[index + 1 :]
                ],
                columns=columns,
            ).to_csv(
                os.path.join(corrected_dir, "race_scalar_paper_compatibility.tsv"),
                sep="\t",
                index=False,
            )
            pd.DataFrame(
                [
                    {
                        "benchmark": "boolq",
                        "pregrouper": "sentence",
                        "scope": "all",
                        "contrast": "canonical",
                        "api_infinity_policy": "drop",
                        "aggregation": "prompt_equal_mean_r2",
                        "model_s": model_s,
                        "model_t": model_t,
                        "metric": "F_align_to_attr",
                        "statistic": "pearson_r2",
                        "f_point": 0.4,
                    }
                    for model_s in reconcile_results.OPEN
                    for model_t in reconcile_results.ALL
                    if model_s != model_t
                ],
                columns=columns,
            ).to_csv(
                os.path.join(corrected_dir, "f_table_prompt_equal_transfer.tsv"),
                sep="\t",
                index=False,
            )
            for filename in (
                "f_table_revision_candidate_pairwise_drop.tsv",
                "f_table_revision_candidate_finite_extreme.tsv",
            ):
                pd.DataFrame(columns=columns).to_csv(
                    os.path.join(corrected_dir, filename), sep="\t", index=False
                )
            raw_path: str = os.path.join(corrected_dir, "bound_raw.tsv")
            with open(raw_path, "w", encoding="utf-8") as output:
                output.write("bound\n")
            raw_hash: str = hashlib.sha256(b"bound\n").hexdigest()
            for filename in reconcile_results.CORRECTED_TABLES:
                table_path_for_sidecar: str = os.path.join(corrected_dir, filename)
                with open(table_path_for_sidecar, "rb") as source:
                    table_hash: str = hashlib.sha256(source.read()).hexdigest()
                with open(
                    f"{table_path_for_sidecar}.provenance.json",
                    "w",
                    encoding="utf-8",
                ) as output:
                    json.dump(
                        {
                            "output": {"sha256": table_hash},
                            "inputs": {
                                "raw": {"path": "bound_raw.tsv", "sha256": raw_hash}
                            },
                        },
                        output,
                    )

            table_path, claims_path, manifest_path, checks_path = (
                reconcile_results.reconcile(
                    archive_dir,
                    corrected_dir,
                    output_dir,
                    allow_unpinned_archive=True,
                )
            )
            ledger: pd.DataFrame = pd.read_csv(table_path, sep="\t")
            claims: pd.DataFrame = pd.read_csv(claims_path, sep="\t")
            self.assertEqual(72, len(claims))
            self.assertEqual(72, claims["claim_id"].nunique())
            dispositions: pd.DataFrame = pd.read_csv(
                os.path.join(output_dir, "manuscript_disposition.tsv"), sep="\t"
            )
            self.assertEqual(
                set(dispositions["issue_id"]),
                {
                    "anli_contrast",
                    "boolq_word_attention_grid",
                    "cross_level_aggregation",
                    "hosted_nonfinite_policy",
                    "main_text_pair_values",
                    "q05_boolq_word_residual",
                    "race_metric_label",
                    "segment_scope",
                },
            )
            self.assertNotIn("-explain", claims.to_csv(index=False))
            self.assertNotIn("-captum", claims.to_csv(index=False))
            archived_word_attention: pd.DataFrame = ledger[
                (ledger["source_state"] == "archive")
                & (ledger["benchmark"] == "boolq")
                & (ledger["pregrouper"] == "word")
                & (ledger["metric"] == "F_attn_mean")
            ]
            self.assertEqual(1, len(archived_word_attention))
            self.assertEqual(
                "reproduced_invalid_sentence_attention_contamination",
                archived_word_attention.iloc[0]["data_status"],
            )
            self.assertEqual(
                "boolq_word_historical_sentence_contaminated_attention",
                archived_word_attention.iloc[0]["segment_grid_id"],
            )
            race_corrected: pd.DataFrame = ledger[
                (ledger["source_state"] == "corrected")
                & (ledger["paper_location"] == "Table 2")
                & (ledger["benchmark"] == "race")
                & (ledger["metric"] == "F_pred")
            ]
            self.assertEqual(1, len(race_corrected))
            self.assertEqual(
                "nonmatching_estimand", race_corrected.iloc[0]["comparison_status"]
            )
            self.assertTrue(pd.isna(race_corrected.iloc[0]["delta_from_published"]))
            self.assertEqual(
                "answer_conditioned_correct_vs_rest",
                race_corrected.iloc[0]["executed_contrast"],
            )
            race_archived: pd.DataFrame = ledger[
                (ledger["source_state"] == "archive")
                & (ledger["paper_location"] == "Table 2")
                & (ledger["benchmark"] == "race")
                & (ledger["metric"] == "F_pred")
            ]
            self.assertEqual(
                "reproduced_metric_mislabeled",
                race_archived.iloc[0]["data_status"],
            )
            self.assertEqual(
                "metric_label_mismatch",
                race_archived.iloc[0]["specification_status"],
            )
            self.assertEqual(
                "differs_from_published_number",
                race_archived.iloc[0]["numerical_reproduction_status"],
            )
            boolq_corrected: pd.DataFrame = ledger[
                (ledger["source_state"] == "corrected")
                & (ledger["paper_location"] == "Table 2")
                & (ledger["benchmark"] == "boolq")
                & (ledger["pregrouper"] == "sentence")
                & (ledger["metric"] == "F_pred")
            ]
            self.assertEqual(
                "nominally_comparable_same_method",
                boolq_corrected.iloc[0]["comparison_status"],
            )
            self.assertEqual(
                "version_not_verifiable",
                boolq_corrected.iloc[0]["model_identity_status"],
            )
            self.assertTrue(pd.isna(boolq_corrected.iloc[0]["delta_from_published"]))
            prompt_equal: pd.DataFrame = ledger[
                (
                    ledger["source_locator"]
                    == "corrected/f_table_prompt_equal_transfer.tsv"
                )
                & (ledger["paper_location"] == "Table 2")
                & (ledger["benchmark"] == "boolq")
                & (ledger["metric"] == "F_align_to_attr")
            ]
            self.assertEqual(1, len(prompt_equal))
            self.assertEqual(
                "nonmatching_estimand", prompt_equal.iloc[0]["comparison_status"]
            )
            self.assertEqual(
                "prompt_equal_mean_r2_then_model_pair_distribution",
                prompt_equal.iloc[0]["aggregation"],
            )

            with open(manifest_path, encoding="utf-8") as source:
                manifest: dict[str, Any] = json.load(source)
            with open(checks_path, encoding="utf-8") as source:
                checks: dict[str, Any] = json.load(source)
            self.assertTrue(manifest["inputs"])
            self.assertTrue(os.path.isfile(claims_path))
            self.assertTrue(
                all(not os.path.isabs(locator) for locator in manifest["inputs"])
            )
            self.assertNotIn(temporary, json.dumps(manifest))
            self.assertNotIn(temporary, json.dumps(checks))
            self.assertIn("136,313-row", manifest["race_caveat"])
            self.assertIn(
                "sentence-level values", manifest["boolq_word_attention_caveat"]
            )
            self.assertFalse(manifest["archive_provenance"]["enforced"])
            self.assertEqual(
                "unversioned_reference_not_byte_pinned",
                manifest["paper_reference"]["identity_status"],
            )
            archive_pin_checks: list[dict[str, Any]] = [
                check
                for check in checks["checks"]
                if check["kind"] == "archive_provenance"
            ]
            self.assertGreaterEqual(len(archive_pin_checks), 33)
            self.assertTrue(all(check["passed"] for check in archive_pin_checks))
            self.assertTrue(any(check["bypassed"] for check in archive_pin_checks))


if __name__ == "__main__":
    unittest.main()
