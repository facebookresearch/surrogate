# Copyright (c) 2025 The Authors
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import json
import os
import tempfile
from types import SimpleNamespace
from contextlib import redirect_stdout
from io import StringIO
from unittest import TestCase
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from benchmark_scripts.f_table import (
    _analysis_jobs,
    _bootstrap_corrs,
    _clamp_api_infinities,
    _emit_pair_corrs,
    _emit_transfer,
    _emit_unavailable_pairs,
    _filter_segment_frame,
    _mask_unsupported_signals,
    _contrast_metadata,
    _derived_input_paths,
    _resolved_contrast,
    _resolved_scope,
    main,
    _per_prompt_completion,
    _per_prompt_logodds,
    _process_benchmark,
    _readout_matches_contrast,
    _unsupported_model_components,
    _uses_layer_readout,
    _weighted_correlation,
    _weighted_ranks,
)


class TestFTable(TestCase):
    def test_cli_help_renders(self) -> None:
        with (
            patch("sys.argv", ["f_table", "--help"]),
            redirect_stdout(StringIO()),
            self.assertRaises(SystemExit) as raised,
        ):
            main()

        self.assertEqual(raised.exception.code, 0)

    def test_analysis_jobs_preserve_explicit_anli_contrasts(self) -> None:
        jobs = _analysis_jobs(
            [("anli_r1", "sentence")],
            ["user"],
            ["canonical", "entailment_neutral", "entailment_contradiction"],
            "entailment_contradiction",
        )

        self.assertEqual(
            jobs,
            [
                (
                    "anli_r1",
                    "sentence",
                    "user",
                    "entailment_contradiction",
                ),
                ("anli_r1", "sentence", "user", "entailment_neutral"),
            ],
        )

    def test_unsupported_components_are_masked_independently(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_dir: str = os.path.join(directory, "lambada", "word")
            os.makedirs(config_dir)
            with open(
                os.path.join(config_dir, "gpt-4o_run.json"),
                "w",
                encoding="utf-8",
            ) as output:
                json.dump(
                    {
                        "availability_status": "unsupported_after_canary",
                        "canary": {
                            "original_coverage": 1.0,
                            "paired_attribution_coverage": 0.2,
                        },
                    },
                    output,
                )
            pd.DataFrame(
                [
                    {
                        "prompt_idx": 0,
                        "original_result_available": True,
                        "segment_result_available": False,
                    },
                    {
                        "prompt_idx": 1,
                        "original_result_available": True,
                        "segment_result_available": False,
                    },
                ]
            ).to_csv(
                os.path.join(config_dir, "gpt-4o_segment.tsv.gz"),
                sep="\t",
                index=False,
            )

            prediction, attribution = _unsupported_model_components(
                directory, "lambada", "word"
            )

            self.assertNotIn("gpt-4o", prediction)
            self.assertIn("gpt-4o", attribution)

    def test_uses_explicit_canonical_logodds(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "model": "model",
                    "prompt_idx": 0,
                    "kind": "orig",
                    "logodds": 1.0,
                    "logodds_contradiction_entailment": -9.0,
                }
            ]
        )

        result = _per_prompt_logodds(frame, "model", "orig")

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.loc[0], 1.0)

    def test_prompt_equal_transfer_averages_correlations_within_prompts(self) -> None:
        index = pd.MultiIndex.from_tuples(
            [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)],
            names=["prompt_idx", "seg_idx"],
        )
        source = pd.Series([0.0, 1.0, 2.0, 100.0, 101.0, 102.0], index=index)
        target = pd.Series([0.0, 1.0, 2.0, 2.0, 1.0, 0.0], index=index)

        rows = _emit_transfer(
            benchmark="test",
            metric="F_test_to_attr",
            src_by_model={"source": source},
            tgt_by_model={"target": target},
            signed=True,
            n_resamples=100,
            conf=0.95,
            rng=np.random.default_rng(42),
            aggregation="prompt_equal_mean_r2",
        )
        points = {row["statistic"]: row["f_point"] for row in rows}

        self.assertAlmostEqual(float(points["spearman"]), 0.0)
        self.assertAlmostEqual(float(points["pearson_r"]), 0.0)
        self.assertAlmostEqual(float(points["pearson_r2"]), 1.0)

    def test_default_transfer_is_row_pooled(self) -> None:
        index = pd.MultiIndex.from_tuples(
            [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)],
            names=["prompt_idx", "seg_idx"],
        )
        source = pd.Series([0.0, 1.0, 2.0, 100.0, 101.0, 102.0], index=index)
        target = pd.Series([0.0, 1.0, 2.0, 2.0, 1.0, 0.0], index=index)

        rows = _emit_transfer(
            benchmark="test",
            metric="F_test_to_attr",
            src_by_model={"source": source},
            tgt_by_model={"target": target},
            signed=True,
            n_resamples=20,
            conf=0.95,
            rng=np.random.default_rng(42),
        )
        points = {row["statistic"]: row["f_point"] for row in rows}

        expected: float = float(np.corrcoef(source, target)[0, 1] ** 2)
        self.assertAlmostEqual(
            float(points["pearson_r"]), float(np.corrcoef(source, target)[0, 1])
        )
        self.assertAlmostEqual(float(points["pearson_r2"]), expected)
        self.assertTrue(all(row["aggregation"] == "row_pooled" for row in rows))

    def test_pair_correlations_report_expected_coverage_and_signed_pearson(
        self,
    ) -> None:
        signals: dict[str, pd.Series] = {
            "a": pd.Series([1.0, 2.0, 3.0, 4.0], index=[0, 1, 2, 3]),
            "b": pd.Series([4.0, np.nan, 2.0, 1.0], index=[0, 1, 2, 3]),
        }

        rows = _emit_pair_corrs(
            "test",
            "F_pred",
            signals,
            20,
            0.95,
            np.random.default_rng(42),
        )
        by_statistic = {row["statistic"]: row for row in rows}

        self.assertEqual(set(by_statistic), {"spearman", "pearson_r", "pearson_r2"})
        pearson = by_statistic["pearson_r"]
        self.assertAlmostEqual(float(pearson["f_point"]), -1.0)
        self.assertEqual(pearson["n_observations"], 3)
        self.assertEqual(pearson["expected_observations"], 4)
        self.assertEqual(pearson["observation_coverage"], 0.75)
        self.assertEqual(pearson["n_prompts"], 3)
        self.assertEqual(pearson["expected_prompts"], 4)
        self.assertEqual(pearson["prompt_coverage"], 0.75)

    def test_transfer_reports_pair_specific_expected_coverage(self) -> None:
        index = pd.MultiIndex.from_tuples(
            [(0, 0), (0, 1), (1, 0), (1, 1)],
            names=["prompt_idx", "seg_idx"],
        )
        source = pd.Series([1.0, 2.0, 3.0, 4.0], index=index)
        target = pd.Series([4.0, np.nan, 2.0, 1.0], index=index)

        rows = _emit_transfer(
            benchmark="test",
            metric="F_test_to_attr",
            src_by_model={"source": source},
            tgt_by_model={"target": target},
            signed=True,
            n_resamples=20,
            conf=0.95,
            rng=np.random.default_rng(42),
        )

        self.assertEqual(
            {row["statistic"] for row in rows},
            {"spearman", "pearson_r", "pearson_r2"},
        )
        for row in rows:
            self.assertEqual(row["n_observations"], 3)
            self.assertEqual(row["expected_observations"], 4)
            self.assertEqual(row["observation_coverage"], 0.75)
            self.assertEqual(row["n_prompts"], 2)
            self.assertEqual(row["expected_prompts"], 2)
            self.assertEqual(row["prompt_coverage"], 1.0)

    def test_clamps_api_infinities_but_not_open_models(self) -> None:
        signal = pd.Series([-float("inf"), -2.0, 3.0, float("inf")])

        api_result = _clamp_api_infinities(signal, "gpt-4o")
        open_result = _clamp_api_infinities(signal, "qwen2.5-7b-instruct")

        self.assertTrue(np.isfinite(api_result).all())
        self.assertAlmostEqual(api_result.iloc[0], -2.03)
        self.assertAlmostEqual(api_result.iloc[-1], 3.04)
        self.assertTrue(np.isinf(open_result.iloc[0]))

    def test_named_contrast_is_selectable(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "model": "model",
                    "prompt_idx": 0,
                    "kind": "orig",
                    "logodds": 1.0,
                    "logodds_entailment_contradiction": 9.0,
                }
            ]
        )

        result = _per_prompt_logodds(
            frame,
            "model",
            "orig",
            "logodds_entailment_contradiction",
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.loc[0], 9.0)

    def test_api_infinities_are_dropped_by_default_and_clamping_is_explicit(
        self,
    ) -> None:
        frame = pd.DataFrame(
            [
                {
                    "model": "gpt-4o",
                    "prompt_idx": 0,
                    "kind": "orig",
                    "logodds": float("inf"),
                },
                {
                    "model": "gpt-4o",
                    "prompt_idx": 1,
                    "kind": "orig",
                    "logodds": 2.0,
                },
            ]
        )

        observed = _per_prompt_logodds(frame, "gpt-4o", "orig")
        sensitivity = _per_prompt_logodds(
            frame,
            "gpt-4o",
            "orig",
            finite_extreme_api=True,
        )

        assert observed is not None
        assert sensitivity is not None
        self.assertTrue(np.isinf(observed.iloc[0]))
        self.assertTrue(np.isfinite(sensitivity).all())

    def test_anli_noncanonical_readout_is_not_relabelled(self) -> None:
        self.assertTrue(_readout_matches_contrast("anli_r1", "canonical"))
        self.assertTrue(_readout_matches_contrast("anli_r1", "entailment_neutral"))
        self.assertFalse(
            _readout_matches_contrast("anli_r1", "entailment_contradiction")
        )
        self.assertTrue(_uses_layer_readout("anli_r1", "entailment_contradiction"))

    def test_resolved_metadata_is_metric_specific(self) -> None:
        self.assertEqual(
            _resolved_contrast("anli_r1", "canonical"),
            "entailment_minus_neutral",
        )
        self.assertEqual(
            _resolved_contrast("anli_r1", "entailment_contradiction"),
            "entailment_minus_contradiction",
        )
        self.assertEqual(_resolved_scope("F_pred", "user"), "prompt_level_full_dialog")
        self.assertEqual(
            _resolved_scope("F_attr", "user"),
            "user_segment_coordinates_from_full_dialog",
        )
        self.assertEqual(
            _contrast_metadata("anli_r1", "entailment_contradiction", "F_attr"),
            (
                "entailment_minus_contradiction",
                "entailment_minus_contradiction",
                "not_applicable",
            ),
        )
        self.assertEqual(
            _contrast_metadata("anli_r1", "entailment_contradiction", "F_align"),
            (
                "entailment_minus_contradiction",
                "entailment_minus_contradiction",
                "entailment_minus_contradiction",
            ),
        )
        self.assertEqual(
            _contrast_metadata(
                "anli_r1", "entailment_contradiction", "F_attn_mean_to_attr"
            ),
            (
                "not_applicable",
                "entailment_minus_contradiction",
                "not_applicable",
            ),
        )

    def test_anli_ec_alignment_uses_final_layer_readout(self) -> None:
        with tempfile.TemporaryDirectory() as results_dir:
            segment_rows: list[dict[str, object]] = []
            logodds_rows: list[dict[str, object]] = []
            for model in ("a", "b"):
                for prompt_idx in range(3):
                    segment_rows.append(
                        {"model": model, "prompt_idx": prompt_idx, "seg_idx": 0}
                    )
                    original: float = float(prompt_idx + 1)
                    logodds_rows.extend(
                        [
                            {
                                "model": model,
                                "prompt_idx": prompt_idx,
                                "seg_idx": np.nan,
                                "kind": "orig",
                                "logodds_entailment_contradiction": original,
                            },
                            {
                                "model": model,
                                "prompt_idx": prompt_idx,
                                "seg_idx": 0,
                                "kind": "ablated",
                                "logodds_entailment_contradiction": (
                                    original - float(prompt_idx + 1)
                                ),
                            },
                        ]
                    )
            pd.DataFrame(segment_rows).to_csv(
                os.path.join(results_dir, "anli_r1_sentence_segments.tsv"),
                sep="\t",
                index=False,
            )
            pd.DataFrame(logodds_rows).to_csv(
                os.path.join(results_dir, "anli_r1_sentence_logodds.tsv"),
                sep="\t",
                index=False,
            )

            def final_readout(
                _results_dir: str,
                _benchmark: str,
                _pregrouper: str,
                model: str,
                _scope: str,
                _contrast: str,
            ) -> SimpleNamespace:
                values: list[float] = [0.0, 1.0, 2.0]
                if model == "b":
                    values = [0.0, 2.0, 1.0]
                return SimpleNamespace(
                    alignment=pd.Series(
                        values,
                        index=pd.MultiIndex.from_tuples(
                            [(idx, 0) for idx in range(3)],
                            names=["prompt_idx", "seg_idx"],
                        ),
                    ),
                    prediction=pd.Series([1.0, 2.0, 3.0], index=range(3)),
                    attribution=pd.Series(
                        [1.0, 2.0, 3.0],
                        index=pd.MultiIndex.from_tuples(
                            [(idx, 0) for idx in range(3)],
                            names=["prompt_idx", "seg_idx"],
                        ),
                    ),
                )

            with (
                patch("benchmark_scripts.f_table.OPEN_MODELS", ("a", "b")),
                patch(
                    "benchmark_scripts.layerwise_fidelity.load_final_layer_readout",
                    side_effect=final_readout,
                ),
            ):
                result = _process_benchmark(
                    "anli_r1",
                    "sentence",
                    results_dir,
                    n_resamples=10,
                    conf=0.95,
                    rng=np.random.default_rng(42),
                    cohort=("a", "b"),
                    scope="all",
                    contrast="entailment_contradiction",
                    bootstrap_seed=42,
                    requested_metrics={"F_align"},
                )

        alignment = [row for row in result if row["metric"] == "F_align"]
        self.assertEqual(len(alignment), 3)
        self.assertTrue(
            all(row["availability_status"] == "available" for row in alignment)
        )
        self.assertTrue(
            all(
                row["readout_contrast"] == "entailment_minus_contradiction"
                for row in alignment
            )
        )

    def test_f_table_provenance_binds_only_used_layer_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as results_dir:
            config_dir: str = os.path.join(results_dir, "anli_r1", "sentence")
            os.makedirs(config_dir)
            model: str = "qwen2.5-7b-instruct"
            filenames: tuple[str, ...] = (
                "segments.tsv.gz",
                f"{model}_segment.tsv.gz",
                f"{model}_run.json",
                f"{model}_layers.tsv.gz",
                f"{model}_layers_run.json",
            )
            for filename in filenames:
                with open(os.path.join(config_dir, filename), "wb") as output:
                    output.write(filename.encode("utf-8"))

            ordinary = _derived_input_paths(
                results_dir,
                [("anli_r1", "sentence")],
                (model,),
            )
            layer_bound = _derived_input_paths(
                results_dir,
                [("anli_r1", "sentence")],
                (model,),
                {("anli_r1", "sentence")},
            )

        self.assertFalse(any("layers" in identifier for identifier in ordinary))
        self.assertTrue(
            any(
                identifier.endswith(f"/{model}_layers.tsv.gz")
                for identifier in layer_bound
            )
        )
        self.assertTrue(
            any(
                identifier.endswith(f"/{model}_layers_run.json")
                for identifier in layer_bound
            )
        )

    def test_cell_keyed_bootstrap_is_invariant_to_unavailable_preceding_pair(
        self,
    ) -> None:
        signals: dict[str, pd.Series] = {
            "missing": pd.Series([np.nan, np.nan, np.nan], index=[0, 1, 2]),
            "a": pd.Series([0.0, 1.0, 3.0], index=[0, 1, 2]),
            "b": pd.Series([0.0, 2.0, 1.0], index=[0, 1, 2]),
        }
        context: tuple[str, ...] = (
            "test",
            "sentence",
            "all_segment_coordinates_from_full_dialog",
            "signal",
            "signal",
            "not_applicable",
            "row_pooled",
        )
        with_missing = _emit_pair_corrs(
            "test",
            "F_attr",
            signals,
            50,
            0.95,
            np.random.default_rng(0),
            bootstrap_seed=42,
            rng_context=context,
        )
        without_missing = _emit_pair_corrs(
            "test",
            "F_attr",
            {"a": signals["a"], "b": signals["b"]},
            50,
            0.95,
            np.random.default_rng(999),
            bootstrap_seed=42,
            rng_context=context,
        )
        retained = [
            row
            for row in with_missing
            if row["model_s"] == "a" and row["model_t"] == "b"
        ]
        self.assertEqual(retained, without_missing)

    def test_unavailable_pair_grid_records_reason(self) -> None:
        rows = _emit_unavailable_pairs(
            "anli_r1",
            "F_align_to_attr",
            ["open-a", "open-b"],
            ["open-a", "hosted"],
            directed=True,
            reason="readout_contrast_mismatch",
        )
        self.assertEqual(len(rows), 9)
        self.assertTrue(
            all(row["availability_status"] == "unavailable" for row in rows)
        )
        self.assertTrue(
            all(
                row["unavailable_reason"] == "readout_contrast_mismatch" for row in rows
            )
        )

    def test_scope_filter_keeps_original_and_selected_ablation_rows(self) -> None:
        frame = pd.DataFrame(
            [
                {"prompt_idx": 0, "seg_idx": np.nan, "kind": "orig"},
                {"prompt_idx": 0, "seg_idx": 0, "kind": "ablated"},
                {"prompt_idx": 0, "seg_idx": 1, "kind": "ablated"},
            ]
        )
        keys = pd.MultiIndex.from_tuples([(0, 1)])

        result = _filter_segment_frame(frame, keys)

        self.assertEqual(result["kind"].tolist(), ["orig", "ablated"])
        self.assertEqual(result["seg_idx"].tolist()[1], 1)

    def test_scope_filter_keeps_orig_for_prompts_without_selected_segments(
        self,
    ) -> None:
        frame = pd.DataFrame(
            [
                {"prompt_idx": 0, "seg_idx": np.nan, "kind": "orig"},
                {"prompt_idx": 0, "seg_idx": 0, "kind": "ablated"},
                {"prompt_idx": 1, "seg_idx": np.nan, "kind": "orig"},
                {"prompt_idx": 1, "seg_idx": 0, "kind": "ablated"},
            ]
        )
        keys = pd.MultiIndex.from_tuples([(0, 0)])

        result = _filter_segment_frame(frame, keys)

        self.assertEqual(result["prompt_idx"].tolist(), [0, 0, 1])
        self.assertEqual(result["kind"].tolist(), ["orig", "ablated", "orig"])

    def test_completion_prediction_scope_keeps_every_sampled_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as results_dir:
            config_dir: str = os.path.join(results_dir, "lambada", "word")
            os.makedirs(config_dir)
            manifest: pd.DataFrame = pd.DataFrame(
                [
                    {
                        "prompt_idx": 0,
                        "seg_idx": 0,
                        "message_role": "system",
                    },
                    {"prompt_idx": 1, "seg_idx": 0, "message_role": "user"},
                    {"prompt_idx": 2, "seg_idx": 0, "message_role": "user"},
                ]
            )
            manifest.to_csv(
                os.path.join(config_dir, "segments.tsv.gz"), sep="\t", index=False
            )
            rows: list[dict[str, object]] = []
            for model, originals in (("a", [0.0, 1.0, 2.0]), ("b", [0.0, 1.0, 4.0])):
                for prompt_idx, original in enumerate(originals):
                    rows.append(
                        {
                            "model": model,
                            "prompt_idx": prompt_idx,
                            "seg_idx": 0,
                            "orig_completion_logprob": original,
                            "ablated_completion_logprob": original - 0.5,
                        }
                    )
            pd.DataFrame(rows).to_csv(
                os.path.join(results_dir, "lambada_word_segments.tsv"),
                sep="\t",
                index=False,
            )

            result = _process_benchmark(
                "lambada",
                "word",
                results_dir,
                n_resamples=10,
                conf=0.95,
                rng=np.random.default_rng(42),
                cohort=("a", "b"),
                scope="user",
            )
            prediction = next(
                row
                for row in result
                if row["metric"] == "F_pred" and row["statistic"] == "pearson_r2"
            )
            self.assertEqual(prediction["n_observations"], 3)
            self.assertEqual(prediction["n_prompts"], 3)

    def test_segment_bootstrap_clusters_by_prompt(self) -> None:
        x = np.array([0.0, 1.0, 10.0, 11.0])
        y = np.array([0.0, 1.0, 11.0, 10.0])
        clusters = np.array([0, 0, 1, 1])

        result = _bootstrap_corrs(
            x,
            y,
            clusters,
            n_resamples=20,
            conf=0.95,
            rng=np.random.default_rng(42),
        )

        self.assertIn("pearson_r2", result)
        self.assertAlmostEqual(result["pearson_r2"][0], np.corrcoef(x, y)[0, 1] ** 2)

    def test_empty_pair_is_materialized_with_missing_statistics(self) -> None:
        result = _bootstrap_corrs(
            np.array([]),
            np.array([]),
            np.array([]),
            n_resamples=20,
            conf=0.95,
            rng=np.random.default_rng(42),
        )

        self.assertEqual(set(result), {"spearman", "pearson_r", "pearson_r2"})
        self.assertTrue(np.isnan(result["pearson_r"][0]))
        self.assertTrue(np.isnan(result["pearson_r2"][0]))

    def test_all_missing_completion_model_remains_in_pair_grid(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "model": "hosted",
                    "prompt_idx": 0,
                    "orig_completion_logprob": np.nan,
                }
            ]
        )

        result = _per_prompt_completion(frame, "hosted")

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.index.tolist(), [0])
        self.assertTrue(np.isnan(result.iloc[0]))

    def test_unsupported_model_is_retained_but_all_values_are_masked(self) -> None:
        supported = pd.Series([1.0, 2.0], index=[0, 1])
        unsupported = pd.Series([3.0, 4.0], index=[0, 1])

        result = _mask_unsupported_signals(
            {"supported": supported, "unsupported": unsupported},
            {"unsupported"},
        )

        self.assertEqual(list(result), ["supported", "unsupported"])
        self.assertTrue(result["supported"].equals(supported))
        self.assertTrue(result["unsupported"].isna().all())
        self.assertEqual(result["unsupported"].index.tolist(), [0, 1])

    def test_weighted_ranks_match_explicit_repetition(self) -> None:
        values = np.array([1.0, 2.0, 1.0])
        weights = np.array([2.0, 1.0, 1.0])

        result = _weighted_ranks(values, weights)

        self.assertEqual(result.tolist(), [2.0, 4.0, 2.0])

    def test_weighted_spearman_matches_explicit_cluster_repetition(self) -> None:
        x = np.array([1.0, 2.0, 1.0, 4.0])
        y = np.array([3.0, 1.0, 3.0, 2.0])
        weights = np.array([2, 1, 3, 2])
        weighted = _weighted_correlation(
            _weighted_ranks(x, weights),
            _weighted_ranks(y, weights),
            weights.astype(float),
        )
        repeated_x = np.repeat(x, weights)
        repeated_y = np.repeat(y, weights)
        explicit = np.corrcoef(rankdata(repeated_x), rankdata(repeated_y))[0, 1]

        self.assertAlmostEqual(weighted, explicit)
