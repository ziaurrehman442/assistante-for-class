import unittest

from app.services.transcript_refiner import (
    TranscriptRefinementError,
    join_turns,
    parse_refinement_response,
    split_turns,
)


class TranscriptStructureTests(unittest.TestCase):
    def test_split_turns_reads_timestamp_speaker_and_text(self):
        turns = split_turns("[00:12] Teacher: hello there\n[00:30] Student 1: hi")
        self.assertEqual(
            turns,
            [("00:12", "Teacher", "hello there"), ("00:30", "Student 1", "hi")],
        )

    def test_join_turns_round_trips(self):
        transcript = "[00:12] Teacher: hello\n[00:30] Student 1: hi"
        self.assertEqual(join_turns(split_turns(transcript)), transcript)

    def test_text_containing_a_colon_is_not_mis_split(self):
        turns = split_turns("[00:05] Teacher: the ratio is a:b and it matters")
        self.assertEqual(len(turns), 1)
        self.assertEqual(turns[0][2], "the ratio is a:b and it matters")


class RefinementParsingTests(unittest.TestCase):
    def test_original_timestamps_and_speakers_are_restored(self):
        original = split_turns("[00:12] Teacher: مرحبا\n[00:30] Student 1: ہاہا")
        refined = "[00:99] WRONG: مرحبا بہت اچھا\n[00:98] WRONG: ہاہا ہاہا"

        rebuilt = parse_refinement_response(refined, original)

        self.assertEqual(
            split_turns(rebuilt),
            [
                ("00:12", "Teacher", "مرحبا بہت اچھا"),
                ("00:30", "Student 1", "ہاہا ہاہا"),
            ],
        )

    def test_changed_turn_count_is_rejected_rather_than_accepted(self):
        original = split_turns("[00:12] Teacher: one\n[00:30] Student 1: two")
        # The model merged both turns into one line.
        with self.assertRaisesRegex(TranscriptRefinementError, "number of turns"):
            parse_refinement_response("[00:12] Teacher: one and two", original)

    def test_dropped_turn_is_rejected(self):
        original = split_turns("[00:12] Teacher: one\n[00:30] Student 1: two")
        with self.assertRaisesRegex(TranscriptRefinementError, "number of turns"):
            parse_refinement_response("[00:12] Teacher: one", original)

    def test_empty_response_is_rejected(self):
        original = split_turns("[00:12] Teacher: one")
        with self.assertRaises(TranscriptRefinementError):
            parse_refinement_response("   \n  ", original)

    def test_blank_refined_turn_is_rejected(self):
        original = split_turns("[00:12] Teacher: one\n[00:30] Student 1: two")
        with self.assertRaises(TranscriptRefinementError):
            parse_refinement_response("[00:12] Teacher: one\n[00:30] Student 1:", original)


if __name__ == "__main__":
    unittest.main()