import unittest

from aimusic.core.config import StyleConfig
from aimusic.core.vocab import DEFAULT_VOCABULARIES, build_default_vocabularies
from aimusic.ml.vocab_export import export_vocabularies_json, vocabularies_from_export


class TestVocabExport(unittest.TestCase):
    def test_default_export_has_spec_field_names(self):
        style = StyleConfig()
        exported = export_vocabularies_json(DEFAULT_VOCABULARIES, style)

        self.assertIn("style_config", exported)
        self.assertEqual(len(exported["meter"]), 4)
        self.assertEqual(len(exported["beat_position"]), 7)
        self.assertEqual(len(exported["boundary"]), 4)
        self.assertEqual(len(exported["key"]), 12)
        self.assertEqual(len(exported["chord"]), 48)
        self.assertEqual(len(exported["role"]), 4)
        self.assertEqual(len(exported["head"]), 8)
        self.assertEqual(len(exported["groove"]), 5)

        meter0 = exported["meter"][0]
        self.assertEqual(set(meter0.keys()), {"id", "label", "beats_per_bar", "strong_beats"})
        chord0 = exported["chord"][0]
        self.assertEqual(set(chord0.keys()), {"id", "label", "root_pc", "quality"})

    def test_export_respects_style_config_sizes(self):
        style = StyleConfig(
            allowed_meters=("4/4",),
            groove_families=("straight",),
            chord_vocabulary_size=24,
            key_vocabulary_size=12,
        )
        vocabs = build_default_vocabularies(style)
        exported = export_vocabularies_json(vocabs, style)

        self.assertEqual(len(exported["chord"]), 24)
        self.assertEqual(exported["style_config"]["chord_vocabulary_size"], 24)
        self.assertEqual(exported["style_config"]["allowed_meters"], ["4/4"])

    def test_exported_ids_resolve_via_token_for_id(self):
        style = StyleConfig()
        exported = export_vocabularies_json(DEFAULT_VOCABULARIES, style)
        for chord_entry in exported["chord"]:
            token = DEFAULT_VOCABULARIES.chords.token_for_id(chord_entry["id"])
            self.assertEqual(token.label, chord_entry["label"])
            self.assertEqual(token.root_pc, chord_entry["root_pc"])

    def test_default_edo_derived_sizes_round_trip_as_none(self):
        style = StyleConfig()
        exported = export_vocabularies_json(DEFAULT_VOCABULARIES, style)

        _, restored = vocabularies_from_export(exported)

        self.assertIsNone(restored.chord_vocabulary_size)
        self.assertIsNone(restored.key_vocabulary_size)


if __name__ == "__main__":
    unittest.main()
