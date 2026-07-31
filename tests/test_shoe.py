"""Tests for blackjack/shoe.py.

These tests ARE the specification. If your implementation makes all of
these pass, it is correct -- you do not need to guess what "correct" means
beyond this file. Run just this file while you work:

    pytest tests/test_shoe.py -v

Read every docstring, not just the assert -- the docstring explains WHY the
test exists, which matters more than the test itself when something fails.
"""

import numpy as np
import pytest

from blackjack.shoe import Shoe, SHOE_SIZE, CARDS_PER_RANK, N_DECKS, PENETRATION
from blackjack.rules import make_rng


def test_shoe_size_is_six_decks():
    """Sanity check on the constant itself, independent of any code you write."""
    assert SHOE_SIZE == 312
    assert N_DECKS == 6


def test_fresh_shoe_has_correct_composition():
    """A freshly built shoe must contain exactly 24 of each rank 1-9 and 96 tens.

    This is the most basic possible check and it is worth running first: if
    this fails, nothing downstream (counting, edge estimation) can be
    trusted, because the whole exercise assumes you know exactly what is in
    the shoe.
    """
    shoe = Shoe(make_rng(0))
    tally = np.bincount(shoe.cards, minlength=11)
    assert np.array_equal(tally, CARDS_PER_RANK)
    assert shoe.cards.size == SHOE_SIZE
    assert shoe.pos == 0


def test_counts_start_at_full_composition():
    shoe = Shoe(make_rng(0))
    assert np.array_equal(shoe.counts, CARDS_PER_RANK)


def test_draw_returns_a_valid_rank():
    shoe = Shoe(make_rng(1))
    for _ in range(50):
        c = shoe.draw()
        assert 1 <= c <= 10


def test_counts_stay_in_sync_with_the_array():
    """After every draw, `counts` must equal what is actually left in `cards`.

    This is the check that would catch a bug where you deal a card but
    forget to decrement its count, or decrement the wrong rank. Such a bug
    would NOT crash anything -- it would just make counting.py's true count
    silently wrong, which is exactly the dangerous kind of bug: no error
    message, just a wrong answer that looks plausible.
    """
    shoe = Shoe(make_rng(2))
    for _ in range(200):
        shoe.draw()
        remaining_by_rank = np.bincount(
            shoe.cards[shoe.pos:], minlength=11)
        assert np.array_equal(shoe.counts, remaining_by_rank), (
            "counts has drifted from the actual remaining cards")


def test_cards_dealt_plus_remaining_is_always_the_shoe_size():
    """The single most important invariant in this file. Assert it, always.

    If this ever fails, it means a card was either duplicated or destroyed
    -- dealt from thin air, or vanished without being counted. That is the
    conservation law of a card game: nothing is created, nothing is
    destroyed, cards only move from "in the shoe" to "dealt".
    """
    shoe = Shoe(make_rng(3))
    for _ in range(250):
        shoe.draw()
        assert shoe.cards_dealt() + shoe.cards_remaining() == SHOE_SIZE


def test_draws_before_reshuffle_are_exactly_without_replacement():
    """Deal the whole shoe (up to penetration) and check no card is reused
    and none is skipped: the sequence dealt must be a permutation of a
    prefix of the shuffled array.
    """
    shoe = Shoe(make_rng(4))
    original = shoe.cards.copy()
    dealt = [shoe.draw() for _ in range(200)]
    assert dealt == original[:200].tolist()


def test_needs_reshuffle_is_false_for_a_fresh_shoe():
    shoe = Shoe(make_rng(5))
    assert not shoe.needs_reshuffle()


def test_reshuffles_at_penetration():
    """After PENETRATION * SHOE_SIZE cards, needs_reshuffle() must flip to True,
    and maybe_reshuffle() must then rebuild a full 312-card shoe.
    """
    shoe = Shoe(make_rng(6))
    threshold = int(PENETRATION * SHOE_SIZE)
    for _ in range(threshold - 1):
        shoe.draw()
    assert not shoe.needs_reshuffle(), (
        f"reshuffling one card too early, at {threshold - 1}/{SHOE_SIZE}")

    shoe.draw()  # this should cross the threshold
    assert shoe.needs_reshuffle()

    fired = shoe.maybe_reshuffle()
    assert fired
    assert shoe.pos == 0
    assert np.array_equal(shoe.counts, CARDS_PER_RANK)


def test_maybe_reshuffle_does_nothing_when_not_due():
    shoe = Shoe(make_rng(7))
    shoe.draw()
    pos_before = shoe.pos
    fired = shoe.maybe_reshuffle()
    assert not fired
    assert shoe.pos == pos_before


def test_probs_sum_to_one_and_match_composition():
    shoe = Shoe(make_rng(8))
    for _ in range(100):
        shoe.draw()
        p = shoe.probs()
        assert p.sum() == pytest.approx(1.0)
        assert np.allclose(p[1:11] * shoe.cards_remaining(), shoe.counts[1:11])


def test_probs_converge_to_the_infinite_deck_limit_early_in_the_shoe():
    """With very few cards dealt, the shoe composition is close to a fresh
    deck, so probs() should be close to rules.CARD_PROBS -- the infinite-deck
    approximation should be excellent when the shoe is nearly full and get
    worse as it empties. This is the mathematical link between Phase 1-2
    (infinite deck) and Phase 3 (finite shoe): the infinite-deck model is
    the limit of the finite-shoe model as N_DECKS -> infinity, or
    equivalently, as cards_dealt -> 0.
    """
    from blackjack.rules import CARD_PROBS
    shoe = Shoe(make_rng(9))
    for _ in range(5):
        shoe.draw()
    assert np.allclose(shoe.probs(), CARD_PROBS, atol=0.01)


def test_probs_reflect_a_low_depleted_shoe():
    """If low cards (2-6) have been disproportionately removed, ten- and
    ace-probability must rise above their infinite-deck values.

    Why this test manipulates `counts` directly instead of just drawing
    randomly: a plain random draw from a shuffled shoe removes every rank in
    the SAME proportion *on average* -- that is what "uniform without
    replacement" means. A single random sequence drifts a little from that
    average only by chance, not systematically. To check the *formula* in
    probs() actually responds to a skewed composition (as opposed to just
    being noisy), we construct the skewed composition directly and check
    the arithmetic, rather than hoping a random shuffle happens to produce
    one. (A *real* skew only shows up over many hands when the running
    count is tracked -- that is exactly what counting.py measures, and
    you'll revisit this exact idea there with real data instead of a
    hand-built scenario.)
    """
    from blackjack.rules import CARD_PROBS
    shoe = Shoe(make_rng(10))
    # Directly simulate "most of the low cards are gone, tens/aces are not":
    # leave only 2 of each low rank (2-6), keep tens and aces near full.
    shoe.counts[2:7] = 2
    shoe.counts[1] = 20
    shoe.counts[7:10] = 20
    shoe.counts[10] = 90
    shoe.pos = SHOE_SIZE - int(shoe.counts.sum())  # keep pos consistent

    assert shoe.probs()[10] > CARD_PROBS[10]
    assert shoe.probs()[1] > CARD_PROBS[1]


def test_composition_conserved_across_many_reshuffles():
    """Run several full shoes back to back; every single one, right after
    reshuffling, must have exactly the canonical composition. This is
    test_fresh_shoe_has_correct_composition again, but stress-tested across
    repeated reshuffles rather than just at construction.
    """
    shoe = Shoe(make_rng(11))
    threshold = int(PENETRATION * SHOE_SIZE)
    for _ in range(5):
        for _ in range(threshold):
            shoe.draw()
        shoe.maybe_reshuffle()
        tally = np.bincount(shoe.cards, minlength=11)
        assert np.array_equal(tally, CARDS_PER_RANK)


def test_no_global_random_state_is_touched():
    """Two shoes built from generators with the same seed must deal
    identical sequences, and building/using a Shoe must not perturb
    numpy's global random state -- everything must flow through the
    Generator passed in, exactly like the rest of this project.
    """
    a = Shoe(make_rng(42))
    b = Shoe(make_rng(42))
    seq_a = [a.draw() for _ in range(300)]
    seq_b = [b.draw() for _ in range(300)]
    assert seq_a == seq_b

    state_before = np.random.get_state()[1].copy()
    Shoe(make_rng(999)).draw()
    state_after = np.random.get_state()[1]
    assert np.array_equal(state_before, state_after), (
        "the global numpy random state changed -- you used np.random.* "
        "or np.random.shuffle somewhere instead of self.rng")
