# tests/test_uk_jurisdiction_verification.py
"""#986: create_case's uk_jurisdiction must survive only when the client's OWN messages
unambiguously name that part of the UK -- never trusted from the model's tool-call argument
alone. See _verified_uk_jurisdiction's docstring for why: two rounds of a stronger prompt
instruction didn't stop the model inventing a value with nothing in the conversation naming a
place."""
import unittest

import the_server as srv


class UkRegionFromTextTests(unittest.TestCase):
    def test_a_named_city_settles_it(self):
        self.assertEqual(srv._uk_region_from_text("I lost my job and I am from Manchester."), "England and Wales")
        self.assertEqual(srv._uk_region_from_text("This happened in Edinburgh."), "Scotland")
        self.assertEqual(srv._uk_region_from_text("I live in Belfast."), "Northern Ireland")

    def test_a_generic_uk_mention_settles_nothing(self):
        self.assertIsNone(srv._uk_region_from_text("I was dismissed from my job in the UK."))

    def test_an_adjective_is_not_a_place(self):
        self.assertIsNone(srv._uk_region_from_text("I'm Scottish but this happened elsewhere."))

    def test_a_city_in_more_than_one_part_of_the_uk_settles_nothing(self):
        self.assertIsNone(srv._uk_region_from_text("I'm not sure, somewhere near Newcastle."))

    def test_naming_two_different_parts_settles_nothing(self):
        self.assertIsNone(srv._uk_region_from_text("I moved from Glasgow to London."))


class VerifiedUkJurisdictionTests(unittest.TestCase):
    def setUp(self):
        self.session_id = "s1"
        srv._context.sessions[self.session_id] = srv.ChatState()

    def tearDown(self):
        srv._context.sessions.pop(self.session_id, None)

    def test_returns_none_when_nothing_was_claimed(self):
        self.assertIsNone(srv._verified_uk_jurisdiction(self.session_id, None))

    def test_confirms_a_claim_the_clients_own_words_back_up(self):
        srv._context.sessions[self.session_id].prompt = ["My landlord in Manchester won't fix the boiler."]
        self.assertEqual(srv._verified_uk_jurisdiction(self.session_id, "England and Wales"), "England and Wales")

    def test_drops_a_claim_with_nothing_in_the_clients_messages_naming_a_place(self):
        srv._context.sessions[self.session_id].prompt = ["I was dismissed from my job in the UK."]
        self.assertIsNone(srv._verified_uk_jurisdiction(self.session_id, "England and Wales"))

    def test_drops_a_claim_that_disagrees_with_what_the_client_actually_said(self):
        srv._context.sessions[self.session_id].prompt = ["This happened in Glasgow."]
        self.assertIsNone(srv._verified_uk_jurisdiction(self.session_id, "England and Wales"))

    def test_never_corroborated_by_the_assistants_own_generated_text(self):
        state = srv._context.sessions[self.session_id]
        state.prompt = ["I was dismissed from my job in the UK."]
        state.generated = ["Sorry to hear that. Was this in Manchester?"]
        self.assertIsNone(srv._verified_uk_jurisdiction(self.session_id, "England and Wales"))

    def test_an_unknown_session_drops_the_claim_rather_than_trusting_it(self):
        self.assertIsNone(srv._verified_uk_jurisdiction("no-such-session", "England and Wales"))


if __name__ == "__main__":
    unittest.main()
