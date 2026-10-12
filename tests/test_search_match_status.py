from types import SimpleNamespace

from backend.app.api.searches import _first_accepted_identity


def test_search_match_status_ignores_unmatched_new_identity():
    unmatched = SimpleNamespace(
        global_person_id=1,
        decision="new_identity",
    )
    result = SimpleNamespace(global_identities=(unmatched,))

    assert _first_accepted_identity(result) is None


def test_search_match_status_uses_only_an_accepted_identity():
    unmatched = SimpleNamespace(
        global_person_id=1,
        decision="new_identity",
    )
    accepted = SimpleNamespace(
        global_person_id=2,
        decision="accept",
    )
    result = SimpleNamespace(global_identities=(unmatched, accepted))

    assert _first_accepted_identity(result) is accepted
