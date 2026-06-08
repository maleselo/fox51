"""Unit tests for Elo model."""

from ml.models.elo import EloModel


def test_home_favorite_has_higher_prob():
    model = EloModel()
    p = model.predict_home_win_prob(home_team_id=1, away_team_id=2)
    assert p > 0.5


def test_ratings_update_after_upset():
    model = EloModel(k_factor=32.0)
    home, away = 10, 20
    before_home = model._rating(home)
    # Home loses as favorite
    for _ in range(5):
        model.update(home, away, home_win=0)
    assert model._rating(home) < before_home
