"""Hybrid move-selection logic for the Battlesnake.

Two engines are combined:

1. A time-boxed, iterative-deepening lookahead search (``choose_move_lookahead``).
   It simulates full turns (all snakes move simultaneously), predicts opponent
   moves with a deterministic one-turn policy, and scores the resulting board
   with a rich evaluation (space, Voronoi territory, tail reachability, food,
   health, hazards, head-to-head risk). The search runs on a lightweight
   immutable board representation (tuples/frozensets) instead of re-copying the
   raw JSON game state at every simulated node, which is what made an
   equivalent fixed-depth search slow. Depth increases turn by turn until a
   wall-clock budget derived from the game's move timeout is exhausted, so the
   bot always returns in time regardless of board size or snake count.
2. A one-ply linear ranking model plus a compact heuristic (unchanged from the
   original fast bot). These serve as an instant fallback if the search raises,
   times out before completing even depth 1, or the snake has no legal move.

Board coordinates: ``(0, 0)`` is the bottom-left corner.
  up    -> y + 1
  down  -> y - 1
  left  -> x - 1
  right -> x + 1

Game-state schema reference: https://docs.battlesnake.com/api
"""

import time
from collections import deque, namedtuple
from typing import Dict, FrozenSet, Iterable, List, Optional, Set, Tuple

Point = Tuple[int, int]

DIRECTIONS: Dict[str, Point] = {
    "up": (0, 1),
    "down": (0, -1),
    "left": (-1, 0),
    "right": (1, 0),
}
MOVE_ORDER: Tuple[str, ...] = ("up", "left", "right", "down")

# Penalty applied to a move that could lose a head-to-head collision.
HEAD_TO_HEAD_PENALTY = 10_000
# Below this health we start actively steering toward food.
HUNGRY_THRESHOLD = 50
DEFAULT_HAZARD_DAMAGE = 14


def get_info() -> Dict[str, str]:
    """Appearance + metadata returned from ``GET /``."""
    return {
        "apiversion": "1",
        "author": "hackathon",
        "color": "#6434eb",
        "head": "smart-caterpillar",
        "tail": "weight",
        "version": "0.3.0-hybrid-lookahead",
    }


def choose_move(game_state: Dict) -> str:
    """Return the next move: time-boxed lookahead, falling back to the model
    and then the heuristic if anything goes wrong or comes up empty."""
    try:
        move = choose_move_lookahead(game_state)
        if move is not None:
            return move
    except Exception:  # noqa: BLE001 - the search must never break gameplay
        pass
    try:
        move = choose_move_model(game_state)
        if move is not None:
            return move
    except Exception:  # noqa: BLE001 - a model issue must never break gameplay
        pass
    return choose_move_heuristic(game_state)


# =============================================================================
# Fast path: one-ply heuristic + linear model (used as instant fallback)
# =============================================================================


def choose_move_heuristic(game_state: Dict) -> str:
    """Return the next move for the current turn."""
    board = game_state["board"]
    you = game_state["you"]
    width: int = board["width"]
    height: int = board["height"]

    head: Point = (you["head"]["x"], you["head"]["y"])
    my_length: int = you["length"]
    health: int = you["health"]

    occupied = _occupied_cells(board["snakes"])
    danger = _head_to_head_cells(board["snakes"], you["id"], my_length)
    foods = [(f["x"], f["y"]) for f in board["food"]]

    best_move = None
    best_score = float("-inf")

    for move, (dx, dy) in DIRECTIONS.items():
        nxt = (head[0] + dx, head[1] + dy)

        if not _in_bounds(nxt, width, height):
            continue
        if nxt in occupied:
            continue

        # Reachable open space from this cell. If we can't fit our own body in
        # the space we'd be moving into, we're about to trap ourselves.
        space = _flood_fill(nxt, occupied, width, height, limit=my_length + 1)
        score = float(space)

        if nxt in danger:
            score -= HEAD_TO_HEAD_PENALTY

        # When hungry, nudge toward the closest food.
        if foods and health < HUNGRY_THRESHOLD:
            nearest = min(_manhattan(nxt, f) for f in foods)
            score += (width + height - nearest) * 2

        if score > best_score:
            best_score = score
            best_move = move

    # No safe move found -> we're cornered. Move up and hope for the best.
    return best_move or "up"


def _is_tail_stacked(snake: Dict) -> bool:
    """True if the snake just ate, so its tail won't move next turn.

    The Battlesnake API duplicates the last body segment for one turn when a
    snake eats (the tail doesn't retract on the turn the snake grows). A
    stacked tail stays occupied next turn; otherwise the tail cell vacates.
    """
    body = snake["body"]
    if len(body) < 2:
        return False
    return body[-1]["x"] == body[-2]["x"] and body[-1]["y"] == body[-2]["y"]


def _occupied_cells(snakes: List[Dict], treat_tails_as_vacating: bool = False) -> Set[Point]:
    """All cells currently filled by any snake's body.

    By default tails are kept occupied too; they only free up *next* turn and
    treating them as solid is the conservative, safe choice for a base bot.
    Pass ``treat_tails_as_vacating=True`` to instead exclude each snake's tail
    cell unless it's stacked (see ``_is_tail_stacked``) — used by lookahead
    evaluation, where the extra precision helps avoid phantom traps.
    """
    occupied: Set[Point] = set()
    for snake in snakes:
        body = snake["body"]
        for seg in body:
            occupied.add((seg["x"], seg["y"]))
        if treat_tails_as_vacating and body and not _is_tail_stacked(snake):
            occupied.discard((body[-1]["x"], body[-1]["y"]))
    return occupied


def _head_to_head_cells(snakes: List[Dict], my_id: str, my_length: int) -> Set[Point]:
    """Cells adjacent to enemy heads that are >= our length.

    Moving onto one of these risks a head-to-head collision we would lose or
    tie, so they are heavily penalized (but not forbidden — sometimes it's the
    only move).
    """
    danger: Set[Point] = set()
    for snake in snakes:
        if snake["id"] == my_id:
            continue
        if snake["length"] < my_length:
            continue
        ehead = (snake["head"]["x"], snake["head"]["y"])
        for dx, dy in DIRECTIONS.values():
            danger.add((ehead[0] + dx, ehead[1] + dy))
    return danger


def _hazard_cells(board: Dict) -> Set[Point]:
    """Cells marked as hazards (extra health cost per turn in hazard game modes)."""
    return {(h["x"], h["y"]) for h in board.get("hazards", [])}


def _flood_fill(start: Point, occupied: Set[Point], width: int, height: int, limit: int) -> int:
    """Count open cells reachable from ``start`` (capped at ``limit``).

    Used to avoid moves that would seal us into a small pocket.
    """
    if start in occupied or not _in_bounds(start, width, height):
        return 0
    seen: Set[Point] = {start}
    stack: List[Point] = [start]
    count = 0
    while stack:
        x, y = stack.pop()
        count += 1
        if count >= limit:
            break
        for dx, dy in DIRECTIONS.values():
            nbr = (x + dx, y + dy)
            if nbr in seen:
                continue
            if not _in_bounds(nbr, width, height):
                continue
            if nbr in occupied:
                continue
            seen.add(nbr)
            stack.append(nbr)
    return count


def _open_neighbor_count(point: Point, blocked: Set[Point], width: int, height: int) -> int:
    return sum(
        1
        for dx, dy in DIRECTIONS.values()
        if _in_bounds((point[0] + dx, point[1] + dy), width, height)
        and (point[0] + dx, point[1] + dy) not in blocked
    )


def _in_bounds(p: Point, width: int, height: int) -> bool:
    return 0 <= p[0] < width and 0 <= p[1] < height


def _manhattan(a: Point, b: Point) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _wall_distance(p: Point, width: int, height: int) -> int:
    return min(p[0], width - 1 - p[0], p[1], height - 1 - p[1])


def _hazard_damage(game_state: Dict) -> int:
    settings = game_state.get("game", {}).get("ruleset", {}).get("settings", {})
    try:
        return int(settings.get("hazardDamagePerTurn", DEFAULT_HAZARD_DAMAGE))
    except (TypeError, ValueError):
        return DEFAULT_HAZARD_DAMAGE


# --- Embedded model features -------------------------------------------------

_BIG = 10_000
_NEIGHBORS = ((0, 1), (0, -1), (-1, 0), (1, 0))


def _bfs_dist(sources, blocked, width, height):
    """Shortest free-cell distances from seed cells."""
    dist = {}
    dq = deque()
    for source in sources:
        if source not in dist:
            dist[source] = 0
            dq.append(source)
    while dq:
        x, y = dq.popleft()
        d = dist[(x, y)]
        for dx, dy in _NEIGHBORS:
            nb = (x + dx, y + dy)
            if 0 <= nb[0] < width and 0 <= nb[1] < height and nb not in blocked and nb not in dist:
                dist[nb] = d + 1
                dq.append(nb)
    return dist


def _candidate_features(state: Dict, move: str) -> Dict[str, float]:
    """Feature vector for playing ``move`` from ``state``. Assumes ``move`` is legal."""
    board = state["board"]
    you = state["you"]
    width, height = board["width"], board["height"]
    head = (you["head"]["x"], you["head"]["y"])
    my_length = you["length"]
    health = you["health"]

    dx, dy = DIRECTIONS[move]
    nxt = (head[0] + dx, head[1] + dy)

    occupied = _occupied_cells(board["snakes"])
    danger = _head_to_head_cells(board["snakes"], you["id"], my_length)
    foods = [(f["x"], f["y"]) for f in board["food"]]
    enemies = [s for s in board["snakes"] if s["id"] != you["id"]]
    enemy_heads = [(s["head"]["x"], s["head"]["y"]) for s in enemies]
    bigger_heads = [(s["head"]["x"], s["head"]["y"]) for s in enemies if s["length"] >= my_length]

    # Voronoi control: cells we reach strictly before any enemy.
    my_dist = _bfs_dist([nxt], occupied, width, height)
    enemy_dist = _bfs_dist(enemy_heads, occupied, width, height) if enemy_heads else {}
    voronoi = sum(1 for cell, md in my_dist.items() if md < enemy_dist.get(cell, _BIG))

    # Tail reachability is a useful anti-self-trap signal.
    my_tail = (you["body"][-1]["x"], you["body"][-1]["y"])
    reach = _bfs_dist([nxt], occupied - {my_tail}, width, height)
    reaches_tail = 1.0 if my_tail in reach else 0.0

    escape = sum(
        1
        for ddx, ddy in _NEIGHBORS
        if _in_bounds((nxt[0] + ddx, nxt[1] + ddy), width, height)
        and (nxt[0] + ddx, nxt[1] + ddy) not in occupied
    )

    nearest_now = min((_manhattan(head, f) for f in foods), default=_BIG)
    nearest_next = min((_manhattan(nxt, f) for f in foods), default=_BIG)
    hungry = health < HUNGRY_THRESHOLD

    return {
        "space_capped": float(_flood_fill(nxt, occupied, width, height, limit=my_length + 1)),
        "open_space": float(_flood_fill(nxt, occupied, width, height, limit=width * height)),
        "voronoi": float(voronoi),
        "reaches_tail": reaches_tail,
        "escape": float(escape),
        "h2h_danger": 1.0 if nxt in danger else 0.0,
        "near_bigger_head": float(min((_manhattan(nxt, h) for h in bigger_heads), default=width + height)),
        "near_enemy_head": float(min((_manhattan(nxt, h) for h in enemy_heads), default=width + height)),
        "wall_dist": float(min(nxt[0], width - 1 - nxt[0], nxt[1], height - 1 - nxt[1])),
        "food_score": float((width + height - nearest_next) * 2) if hungry and foods else 0.0,
        "food_delta": float(nearest_now - nearest_next) if foods else 0.0,
        "is_food": 1.0 if nxt in foods else 0.0,
        "dist_to_center": abs(nxt[0] - (width - 1) / 2) + abs(nxt[1] - (height - 1) / 2),
    }


# --- Model -----------------------------------------------------
# Embedded standardized linear model.

_MODEL: Dict = {
    "feature_names": [
        "space_capped",
        "open_space",
        "voronoi",
        "reaches_tail",
        "escape",
        "h2h_danger",
        "near_bigger_head",
        "near_enemy_head",
        "wall_dist",
        "food_score",
        "food_delta",
        "is_food",
        "dist_to_center",
    ],
    "mean": [
        7.357954545454546,
        100.9034090909091,
        48.26988636363637,
        0.9943181818181818,
        2.4431818181818183,
        0.04261363636363636,
        9.673295454545455,
        4.676136363636363,
        1.625,
        0.8920454545454546,
        0.14772727272727273,
        0.036931818181818184,
        5.056818181818182,
    ],
    "std": [
        3.5995966185276513,
        22.80542174802676,
        31.41119158524981,
        0.07516338951888041,
        0.6235520417417705,
        0.20198444088469822,
        7.9675173248507924,
        2.2532045017839604,
        1.3552297691803878,
        5.861056404757769,
        0.9449599886584031,
        0.18859442989548575,
        2.34451950177747,
    ],
    "coef": [
        0.00010539398521136327,
        -1.6778512168946185,
        80.89420182766183,
        9.793855564450467,
        0.7884630868036275,
        -11.025170822665032,
        -0.7981723553489,
        0.5410534990053248,
        1.5629078731518526,
        7.582325762611304,
        0.12463070008097832,
        0.21036618806863483,
        1.836259515524985,
    ],
    "intercept": 0.0,
    "top1_accuracy": 0.9928571428571429,
}


def choose_move_model(game_state: Dict) -> Optional[str]:
    """Score each legal move with the trained model; return the best.

    Returns ``None`` (so the caller falls back to the heuristic) if the model
    isn't available or the snake is trapped with no legal move.
    """
    legal = _legal_moves(game_state)
    if not legal:
        return None

    names = _MODEL["feature_names"]
    mean = _MODEL["mean"]
    std = _MODEL["std"]
    coef = _MODEL["coef"]
    intercept = _MODEL["intercept"]

    best_move, best_score = None, float("-inf")
    for move in legal:
        feats = _candidate_features(game_state, move)
        score = intercept
        for i, name in enumerate(names):
            z = (feats.get(name, 0.0) - mean[i]) / std[i] if std[i] else 0.0
            score += coef[i] * z
        if score > best_score:
            best_score, best_move = score, move
    return best_move


def _legal_moves(game_state: Dict) -> List[str]:
    board = game_state["board"]
    width, height = board["width"], board["height"]
    head = (game_state["you"]["head"]["x"], game_state["you"]["head"]["y"])
    occupied = _occupied_cells(board["snakes"])
    return [
        move
        for move, (dx, dy) in DIRECTIONS.items()
        if _in_bounds((head[0] + dx, head[1] + dy), width, height)
        and (head[0] + dx, head[1] + dy) not in occupied
    ]


# =============================================================================
# Smart path: time-boxed, iterative-deepening simultaneous-turn lookahead
# =============================================================================
#
# Board state here is an immutable tuple/frozenset representation, built once
# per ``choose_move`` call and threaded through the search without ever being
# deep-copied. This is what lets the search go several plies deep inside a
# normal move-timeout budget; the original dict+deepcopy simulator this was
# ported from spent most of its time copying JSON structures.

DEATH_SCORE = -1_000_000_000.0
WIN_SCORE = 100_000_000.0
MAX_SEARCH_DEPTH = 30
DEFAULT_TIMEOUT_MS = 500
TIME_SAFETY_MARGIN_MS = 120
# Flood-fill cap used while predicting opponent moves inside the search tree.
# Exact large-scale space counts don't matter for "is this safe", only
# distinguishing cramped vs. open, so a small cap keeps inner nodes cheap.
STANDARD_MOVE_FLOOD_CAP = 60

# The opponent policy is a single deterministic guess. Trusting it blindly is
# risky right next to a snake that could instead attack or cut us off, so for
# the closest nearby threat we hedge: branch over its top-K plausible moves
# for the first few plies and assume it picks whichever is worst for us. This
# is limited to one enemy and a couple of plies so the branching stays cheap.
ADVERSARIAL_RADIUS = 4
ADVERSARIAL_PLIES = 2
ENEMY_BRANCH_TOP_K = 2

# Leaf-evaluation bonus for squeezing a nearby enemy's reachable space down
# toward (or below) its own length -- i.e. rewarding moves that we predict
# will box an opponent into a bad spot, not just moves that are safe for us.
TRAP_CHECK_RADIUS = 4
TRAP_SPACE_SLACK = 2
TRAP_FLOOD_MARGIN = 4

SnakeState = namedtuple("SnakeState", "id body health")


class Board:
    """Lightweight immutable board snapshot used only by the lookahead search."""

    __slots__ = ("width", "height", "snakes", "food", "hazards", "turn")

    def __init__(
        self,
        width: int,
        height: int,
        snakes: Tuple[SnakeState, ...],
        food: FrozenSet[Point],
        hazards: FrozenSet[Point],
        turn: int,
    ) -> None:
        self.width = width
        self.height = height
        self.snakes = snakes
        self.food = food
        self.hazards = hazards
        self.turn = turn


class TimeUp(Exception):
    """Raised to unwind the search once the move-time budget is exhausted."""


def choose_move_lookahead(game_state: Dict) -> Optional[str]:
    """Search several turns ahead, predicting opponents with a fixed policy.

    Runs iterative deepening (depth 1, 2, 3, ...) until ``MAX_SEARCH_DEPTH`` or
    a wall-clock deadline derived from the game's move timeout is reached,
    returning the best move found by the deepest depth that finished in time.
    """
    board = _parse_board(game_state)
    my_id = game_state["you"]["id"]
    if _find(board, my_id) is None:
        return None

    candidates = _candidate_moves(board, my_id)
    if not candidates:
        return None

    timeout_ms = game_state.get("game", {}).get("timeout") or DEFAULT_TIMEOUT_MS
    try:
        timeout_ms = int(timeout_ms)
    except (TypeError, ValueError):
        timeout_ms = DEFAULT_TIMEOUT_MS
    budget = max(0.05, (timeout_ms - TIME_SAFETY_MARGIN_MS) / 1000.0)
    deadline = time.monotonic() + budget

    hazard_damage = _hazard_damage(game_state)
    initial_enemy_count = max(0, len(board.snakes) - 1)

    # Order candidates with the cheap one-ply policy so a mid-depth timeout
    # still leaves us with a sensible move, and shallow depths look decent.
    move_order = sorted(
        candidates,
        key=lambda m: _score_standard_move(board, my_id, m, hazard_damage, STANDARD_MOVE_FLOOD_CAP),
        reverse=True,
    )
    best_move = move_order[0]

    # (state, depth) -> value. Values are depth-remaining specific, so entries
    # computed at a shallow iterative-deepening pass stay valid and get reused
    # by deeper passes recursing into the same subtrees.
    cache: Dict[Tuple, float] = {}

    depth = 1
    while depth <= MAX_SEARCH_DEPTH and time.monotonic() < deadline:
        radius = min(4 + 2 * depth, board.width + board.height)
        relevant_ids = _relevant_enemy_ids(board, my_id, radius)
        try:
            move, _score = _search_root(
                board, my_id, depth, initial_enemy_count, hazard_damage,
                relevant_ids, deadline, cache, move_order,
            )
        except TimeUp:
            break

        if move is not None:
            best_move = move
            move_order = [move] + [m for m in move_order if m != move]
        depth += 1

    return best_move


def _search_root(
    board: Board,
    my_id: str,
    depth: int,
    initial_enemy_count: int,
    hazard_damage: int,
    relevant_ids: Set[str],
    deadline: float,
    cache: Dict[Tuple, float],
    move_order: Iterable[str],
    plies_from_root: int = 0,
) -> Tuple[Optional[str], float]:
    enemy_options = _predict_enemy_move_options(board, my_id, hazard_damage, relevant_ids, plies_from_root)
    best_move: Optional[str] = None
    best_score = DEATH_SCORE

    for move in move_order:
        _check_time(deadline)
        worst = float("inf")
        for enemy_moves in _enemy_move_combos(enemy_options):
            moves = dict(enemy_moves)
            moves[my_id] = move
            nxt = _simulate(board, moves, hazard_damage)

            if _find(nxt, my_id) is None:
                value = DEATH_SCORE
            elif depth == 1:
                value = _evaluate(nxt, my_id, initial_enemy_count)
            else:
                value = _search_value(
                    nxt, my_id, depth - 1, initial_enemy_count, hazard_damage,
                    relevant_ids, deadline, cache, plies_from_root + 1,
                )
            if value < worst:
                worst = value

        score = worst + _root_tiebreak(board, my_id, move, hazard_damage)

        if score > best_score:
            best_score = score
            best_move = move

    return best_move, best_score


def _search_value(
    board: Board,
    my_id: str,
    depth: int,
    initial_enemy_count: int,
    hazard_damage: int,
    relevant_ids: Set[str],
    deadline: float,
    cache: Dict[Tuple, float],
    plies_from_root: int,
) -> float:
    _check_time(deadline)

    if _find(board, my_id) is None:
        return DEATH_SCORE
    if depth <= 0:
        return _evaluate(board, my_id, initial_enemy_count)

    # Adversarial hedging only depends on whether we're still within the first
    # few plies of the root, so cap it in the cache key rather than using the
    # exact ply count -- keeps transposition hits working across the many
    # (board, depth) pairs reached beyond that threshold.
    key = (_state_key(board), depth, min(plies_from_root, ADVERSARIAL_PLIES))
    cached = cache.get(key)
    if cached is not None:
        return cached

    enemy_options = _predict_enemy_move_options(board, my_id, hazard_damage, relevant_ids, plies_from_root)
    best = DEATH_SCORE

    for move in _candidate_moves(board, my_id):
        worst = float("inf")
        for enemy_moves in _enemy_move_combos(enemy_options):
            moves = dict(enemy_moves)
            moves[my_id] = move
            nxt = _simulate(board, moves, hazard_damage)

            if _find(nxt, my_id) is None:
                value = DEATH_SCORE
            elif depth == 1:
                value = _evaluate(nxt, my_id, initial_enemy_count)
            else:
                value = _search_value(
                    nxt, my_id, depth - 1, initial_enemy_count, hazard_damage,
                    relevant_ids, deadline, cache, plies_from_root + 1,
                )
            if value < worst:
                worst = value

        if worst > best:
            best = worst

    cache[key] = best
    return best


def _check_time(deadline: float) -> None:
    if time.monotonic() > deadline:
        raise TimeUp()


def _root_tiebreak(board: Board, sid: str, move: str, hazard_damage: int) -> float:
    """Small one-turn preference used only when rollouts are effectively tied."""
    standard = _score_standard_move(board, sid, move, hazard_damage, STANDARD_MOVE_FLOOD_CAP)
    if standard <= DEATH_SCORE / 2:
        return -0.004
    return max(-0.003, min(0.003, standard / 1_000_000_000.0))


# --- Deterministic one-turn opponent policy ---------------------------------


def _relevant_enemy_ids(board: Board, my_id: str, radius: int) -> Set[str]:
    """Enemies close enough to matter within the current search radius.

    Distant snakes are assumed to keep going straight (see
    ``_predict_enemy_moves``) instead of running the full one-ply policy on
    them, which is the dominant per-node cost in the search.
    """
    me = _find(board, my_id)
    if me is None:
        return set()
    head = me.body[0]
    return {s.id for s in board.snakes if s.id != my_id and _manhattan(head, s.body[0]) <= radius}


def _nearest_threat_enemy(board: Board, my_id: str, radius: int) -> Optional[str]:
    """The single closest enemy within ``radius``, if any -- the one whose
    predicted move we can least afford to get wrong."""
    me = _find(board, my_id)
    if me is None:
        return None
    head = me.body[0]
    best_id: Optional[str] = None
    best_dist = radius + 1
    for s in board.snakes:
        if s.id == my_id:
            continue
        dist = _manhattan(head, s.body[0])
        if dist <= radius and dist < best_dist:
            best_dist = dist
            best_id = s.id
    return best_id


def _predict_enemy_move_options(
    board: Board,
    my_id: str,
    hazard_damage: int,
    relevant_ids: Set[str],
    plies_from_root: int,
) -> Dict[str, List[str]]:
    """One plausible move per enemy, except the nearest close threat during the
    first few plies, which gets its top few plausible moves so the search can
    hedge against us guessing its reaction wrong."""
    adversarial_id = (
        _nearest_threat_enemy(board, my_id, ADVERSARIAL_RADIUS)
        if plies_from_root < ADVERSARIAL_PLIES
        else None
    )

    result: Dict[str, List[str]] = {}
    for s in board.snakes:
        if s.id == my_id:
            continue
        if s.id == adversarial_id:
            result[s.id] = _top_k_standard_moves(board, s.id, hazard_damage, ENEMY_BRANCH_TOP_K)
            continue
        if s.id in relevant_ids:
            result[s.id] = [_choose_standard_move(board, s.id, hazard_damage) or "up"]
            continue
        # Cheap fallback for irrelevant snakes: keep going straight if that
        # stays in-bounds, else grab any in-bounds move. No flood fill.
        forward = _current_direction(s.body)
        if forward is not None:
            dx, dy = DIRECTIONS[forward]
            nxt = (s.body[0][0] + dx, s.body[0][1] + dy)
            if _in_bounds(nxt, board.width, board.height):
                result[s.id] = [forward]
                continue
        cand = _candidate_moves(board, s.id)
        result[s.id] = [cand[0]] if cand else ["up"]
    return result


def _enemy_move_combos(options: Dict[str, List[str]]) -> Iterable[Dict[str, str]]:
    """Expand per-enemy move options into concrete move assignments.

    At most one enemy ever has more than one option (the adversarial hedge),
    so this only ever yields 1 or ``ENEMY_BRANCH_TOP_K`` combinations -- never
    a full cartesian blow-up across all enemies.
    """
    varying = [(sid, moves) for sid, moves in options.items() if len(moves) > 1]
    base = {sid: moves[0] for sid, moves in options.items() if len(moves) == 1}
    if not varying:
        yield base
        return
    sid, moves = varying[0]
    for move in moves:
        combo = dict(base)
        combo[sid] = move
        yield combo


def _choose_standard_move(board: Board, sid: str, hazard_damage: int) -> Optional[str]:
    best_move: Optional[str] = None
    best_score = DEATH_SCORE
    for move in _candidate_moves(board, sid):
        score = _score_standard_move(board, sid, move, hazard_damage, STANDARD_MOVE_FLOOD_CAP)
        if score > best_score:
            best_score = score
            best_move = move
    return best_move


def _top_k_standard_moves(board: Board, sid: str, hazard_damage: int, k: int) -> List[str]:
    scored = [
        (move, _score_standard_move(board, sid, move, hazard_damage, STANDARD_MOVE_FLOOD_CAP))
        for move in _candidate_moves(board, sid)
    ]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    moves = [move for move, _score in scored[:k]]
    return moves or ["up"]


def _score_standard_move(board: Board, sid: str, move: str, hazard_damage: int, flood_cap: int) -> float:
    s = _find(board, sid)
    if s is None:
        return DEATH_SCORE

    body = s.body
    head = body[0]
    dx, dy = DIRECTIONS[move]
    nxt = (head[0] + dx, head[1] + dy)
    if not _in_bounds(nxt, board.width, board.height):
        return DEATH_SCORE

    eats = nxt in board.food
    projected_length = len(body) + (1 if eats else 0)

    blocked = _occupied_without_vacating_tails(board.snakes)
    if eats:
        blocked.add(body[-1])
    if nxt in blocked:
        return DEATH_SCORE + 1_000.0

    projected_occupied = blocked - {nxt}
    space = _flood_fill(nxt, projected_occupied, board.width, board.height, flood_cap)
    mobility = _open_neighbor_count(nxt, projected_occupied, board.width, board.height)
    score = float(space * 120 + mobility * 45)

    for enemy in board.snakes:
        if enemy.id == sid:
            continue
        enemy_head = enemy.body[0]
        if _manhattan(nxt, enemy_head) != 1:
            continue
        if len(enemy.body) >= projected_length:
            score -= 200_000.0
        else:
            score += 3_000.0

    health_after = s.health - 1
    if eats:
        health_after = 100
    if nxt in board.hazards:
        health_after -= hazard_damage
        score -= 4_000.0 + max(0, 35 - health_after) * 500.0
    if health_after <= 0:
        return DEATH_SCORE + 2_000.0

    if board.food:
        distance = min(_manhattan(nxt, food) for food in board.food)
        hunger = max(0, HUNGRY_THRESHOLD - s.health)
        score -= distance * (20.0 + hunger * 6.0)
        if eats:
            score += 10_000.0 + hunger * 300.0

    forward = _current_direction(body)
    if forward == move:
        score += 3.0

    score += _wall_distance(nxt, board.width, board.height) * 2.0
    return score


# --- Turn simulator ----------------------------------------------------------


def _simulate(board: Board, moves: Dict[str, str], hazard_damage: int) -> Board:
    """Resolve one simultaneous Battlesnake turn and return a new ``Board``."""
    moved: List[SnakeState] = []
    consumed: Set[Point] = set()

    for s in board.snakes:
        move = moves.get(s.id) or _current_direction(s.body) or "up"
        if move not in DIRECTIONS:
            move = _current_direction(s.body) or "up"
        dx, dy = DIRECTIONS[move]
        head = s.body[0]
        new_head = (head[0] + dx, head[1] + dy)
        new_body = (new_head,) + s.body[:-1]
        health = s.health - 1

        if new_head in board.food:
            health = 100
            new_body = new_body + (s.body[-1],)
            consumed.add(new_head)
        if new_head in board.hazards:
            health -= hazard_damage

        moved.append(SnakeState(s.id, new_body, health))

    dead: Set[str] = set()

    for s in moved:
        if not _in_bounds(s.body[0], board.width, board.height) or s.health <= 0:
            dead.add(s.id)

    heads: Dict[Point, List[SnakeState]] = {}
    for s in moved:
        if s.id in dead:
            continue
        heads.setdefault(s.body[0], []).append(s)

    for group in heads.values():
        if len(group) < 2:
            continue
        longest = max(len(s.body) for s in group)
        winners = [s for s in group if len(s.body) == longest]
        if len(winners) == 1:
            dead.update(s.id for s in group if s.id != winners[0].id)
        else:
            dead.update(s.id for s in group)

    body_cells: Set[Point] = set()
    for s in moved:
        body_cells.update(s.body[1:])

    for s in moved:
        if s.id in dead:
            continue
        if s.body[0] in body_cells:
            dead.add(s.id)

    survivors = tuple(s for s in moved if s.id not in dead)
    new_food = frozenset(board.food - consumed)
    return Board(board.width, board.height, survivors, new_food, board.hazards, board.turn + 1)


# --- Terminal board evaluation ------------------------------------------------


def _evaluate(board: Board, my_id: str, initial_enemy_count: int) -> float:
    me = _find(board, my_id)
    if me is None:
        return DEATH_SCORE

    enemies = [s for s in board.snakes if s.id != my_id]
    if not enemies:
        return WIN_SCORE + me.health * 1_000.0 + len(me.body) * 10_000.0

    my_head = me.body[0]
    blocked = _occupied_without_vacating_tails(board.snakes)
    blocked.discard(my_head)

    space = _flood_fill(my_head, blocked, board.width, board.height, board.width * board.height)
    mobility = _open_neighbor_count(my_head, blocked, board.width, board.height)
    territory = _voronoi_territory(board, my_id)

    eliminated = initial_enemy_count - len(enemies)
    score = 0.0
    score += eliminated * 250_000.0
    score += space * 1_000.0
    score += territory * 350.0
    score += mobility * 2_500.0
    score += len(me.body) * 5_000.0
    score += me.health * 120.0
    score += _wall_distance(my_head, board.width, board.height) * 50.0

    tail = me.body[-1]
    tail_blocked = set(blocked)
    if _tail_vacates(me.body):
        tail_blocked.discard(tail)
    tail_reach = _bfs_dist([my_head], tail_blocked, board.width, board.height)
    if tail in tail_reach:
        score += 12_000.0
    else:
        score -= 20_000.0

    if board.food:
        food_distance = min(_manhattan(my_head, food) for food in board.food)
        if me.health < HUNGRY_THRESHOLD:
            score -= food_distance * (HUNGRY_THRESHOLD - me.health + 10) * 120.0
        else:
            score -= food_distance * 20.0

    if my_head in board.hazards:
        score -= 15_000.0

    for enemy in enemies:
        enemy_head = enemy.body[0]
        distance = _manhattan(my_head, enemy_head)
        if distance == 1:
            if len(enemy.body) >= len(me.body):
                score -= 100_000.0
            else:
                score += 15_000.0

        # Reward predicting/inducing a cramped spot for a nearby enemy,
        # independent of whether our search horizon reaches the turn it dies.
        if distance <= TRAP_CHECK_RADIUS:
            score += _trap_bonus(board, enemy)

    return score


def _trap_bonus(board: Board, enemy: SnakeState) -> float:
    """Bonus for a nearby enemy whose reachable space is running out relative
    to its own length -- a proxy for "we predict/are inducing it into a bad
    spot" that doesn't require the trap to fully resolve within the search
    horizon."""
    enemy_head = enemy.body[0]
    blocked = _occupied_without_vacating_tails(board.snakes)
    blocked.discard(enemy_head)
    space = _flood_fill(enemy_head, blocked, board.width, board.height, len(enemy.body) + TRAP_FLOOD_MARGIN)
    slack = space - len(enemy.body)
    if slack > TRAP_SPACE_SLACK:
        return 0.0
    return (TRAP_SPACE_SLACK - slack + 1) * 4_000.0


def _voronoi_territory(board: Board, my_id: str) -> int:
    """Count cells reached strictly sooner by us than by any opponent."""
    me = _find(board, my_id)
    if me is None:
        return 0

    blocked = _occupied_without_vacating_tails(board.snakes)
    heads = {s.body[0] for s in board.snakes}
    blocked.difference_update(heads)

    my_head = me.body[0]
    enemy_heads = [s.body[0] for s in board.snakes if s.id != my_id]
    my_dist = _bfs_dist([my_head], blocked, board.width, board.height)
    enemy_dist = _bfs_dist(enemy_heads, blocked, board.width, board.height) if enemy_heads else {}

    infinity = board.width * board.height + 1
    return sum(1 for cell, dist in my_dist.items() if dist < enemy_dist.get(cell, infinity))


# --- Board helpers -------------------------------------------------------


def _parse_board(game_state: Dict) -> Board:
    b = game_state["board"]
    snakes = tuple(
        SnakeState(
            id=s["id"],
            body=tuple((seg["x"], seg["y"]) for seg in s["body"]),
            health=int(s.get("health", 100)),
        )
        for s in b["snakes"]
    )
    food = frozenset((f["x"], f["y"]) for f in b.get("food", []))
    hazards = frozenset((h["x"], h["y"]) for h in b.get("hazards", []))
    return Board(int(b["width"]), int(b["height"]), snakes, food, hazards, int(game_state.get("turn", 0)))


def _find(board: Board, sid: str) -> Optional[SnakeState]:
    for s in board.snakes:
        if s.id == sid:
            return s
    return None


def _candidate_moves(board: Board, sid: str) -> List[str]:
    """In-bounds moves for a snake; collision legality is resolved by simulation."""
    s = _find(board, sid)
    if s is None:
        return []
    head = s.body[0]
    moves = [m for m, (dx, dy) in DIRECTIONS.items() if _in_bounds((head[0] + dx, head[1] + dy), board.width, board.height)]
    return moves or list(DIRECTIONS.keys())


def _current_direction(body: Tuple[Point, ...]) -> Optional[str]:
    """Infer current travel direction from the first non-overlapping segment."""
    if len(body) < 2:
        return None
    head = body[0]
    for seg in body[1:]:
        if seg == head:
            continue
        delta = (head[0] - seg[0], head[1] - seg[1])
        for move, vector in DIRECTIONS.items():
            if delta == vector:
                return move
        break
    return None


def _tail_vacates(body: Tuple[Point, ...]) -> bool:
    return len(body) < 2 or body[-1] != body[-2]


def _occupied_without_vacating_tails(snakes: Tuple[SnakeState, ...]) -> Set[Point]:
    occupied: Set[Point] = set()
    for s in snakes:
        occupied.update(s.body)
        if s.body and _tail_vacates(s.body):
            occupied.discard(s.body[-1])
    return occupied


def _state_key(board: Board) -> Tuple:
    snakes = tuple(sorted((s.id, s.health, s.body) for s in board.snakes))
    return (board.turn, snakes, tuple(sorted(board.food)), tuple(sorted(board.hazards)))
