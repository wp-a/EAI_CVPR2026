import argparse
import json
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class VHObject:
    name: str
    object_id: str
    properties: frozenset

    @property
    def key(self):
        return (self.name, self.object_id)


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def section(text, start, end):
    return text.split(start, 1)[1].split(end, 1)[0]


def parse_properties(raw):
    raw = raw.strip()
    if not raw.startswith("["):
        return frozenset()
    return frozenset(re.findall(r"'([^']+)'", raw))


def parse_prompt(prompt):
    object_block = section(
        prompt,
        "Objects in the scene:\n",
        "-----------------",
    )
    current_block = section(
        prompt,
        "The current environment state is\n",
        "Node goals are:",
    )
    node_goal_block = section(
        prompt,
        "Node goals are:\n",
        "Edge goals are:",
    )
    edge_goal_block = section(
        prompt,
        "Edge goals are:\n",
        "Action goals are:",
    )
    action_goal_block = section(
        prompt,
        "Action goals are:\n",
        "Please output",
    )

    objects = []
    for line in object_block.splitlines():
        line = line.strip()
        match = re.match(r"([^,]+), id: ([^,]+), properties: (.*)$", line)
        if match:
            objects.append(
                VHObject(
                    match.group(1).strip(),
                    match.group(2).strip(),
                    parse_properties(match.group(3)),
                )
            )

    nodes = []
    in_nodes = False
    for line in current_block.splitlines():
        stripped = line.strip()
        if stripped == "Nodes:":
            in_nodes = True
            continue
        if stripped == "Edges:":
            in_nodes = False
            continue
        if in_nodes:
            match = re.match(r"([^,]+), states: \[(.*?)\], properties:\[(.*?)\]", stripped)
            if match:
                states = frozenset(re.findall(r"'([^']+)'", match.group(2)))
                props = frozenset(re.findall(r"'([^']+)'", match.group(3)))
                nodes.append((match.group(1).strip(), states, props))

    edges = []
    for line in current_block.splitlines():
        match = re.match(
            r"<(.+?)> \((.+?)\) is ([A-Z_]+) to <(.+?)> \((.+?)\)",
            line.strip(),
        )
        if match:
            edges.append(
                (
                    match.group(1).strip(),
                    match.group(2).strip(),
                    match.group(3).strip(),
                    match.group(4).strip(),
                    match.group(5).strip(),
                )
            )

    node_goals = []
    for line in node_goal_block.splitlines():
        match = re.match(r"(.+?) is ([A-Z_]+)$", line.strip())
        if match:
            node_goals.append((match.group(1).strip(), match.group(2).strip()))

    edge_goals = []
    for line in edge_goal_block.splitlines():
        match = re.match(r"(.+?) is ([A-Z_]+) to (.+?)$", line.strip())
        if match:
            edge_goals.append(
                (match.group(1).strip(), match.group(2).strip(), match.group(3).strip())
            )

    action_goals = []
    for line in action_goal_block.splitlines():
        line = line.strip()
        if not line or line.startswith("-") or "no action requirement" in line:
            continue
        if line.startswith("The following action"):
            continue
        action_goals.append([part.strip() for part in re.split(r"\s+or\s+", line)])

    return {
        "objects": objects,
        "nodes": nodes,
        "edges": edges,
        "node_goals": node_goals,
        "edge_goals": edge_goals,
        "action_goals": action_goals,
    }


class VirtualHomeCompiler:
    def __init__(self, parsed):
        self.objects = parsed["objects"]
        self.node_goals = parsed["node_goals"]
        self.edge_goals = parsed["edge_goals"]
        self.action_goals = parsed["action_goals"]

        self.by_name = defaultdict(list)
        for obj in self.objects:
            self.by_name[obj.name].append(obj)

        self.actions = []
        self.near = set()
        self.held = set()
        self.states = defaultdict(set)
        self.locations = {}
        self.character_states = set()

        for obj_name, states, _props in parsed["nodes"]:
            matches = self.by_name.get(obj_name, [])
            if obj_name == "character":
                self.character_states.update(states)
            for obj in matches:
                self.states[obj.key].update(states)

        for src_name, src_id, rel, dst_name, dst_id in parsed["edges"]:
            src = self.object_by_name_id(src_name, src_id)
            dst = self.object_by_name_id(dst_name, dst_id)
            if not src or not dst:
                continue
            if src.name == "character":
                if rel in {"NEAR", "CLOSE", "FACING"}:
                    self.near.add(dst.key)
                if rel in {"HOLDS_RH", "HOLDS_LH"}:
                    self.held.add(dst.key)
            elif rel in {"INSIDE", "ON"}:
                self.locations[src.key] = (rel, dst.key)

    def object_by_name_id(self, name, object_id):
        for obj in self.by_name.get(name, []):
            if obj.object_id == object_id:
                return obj
        return None

    def first(self, name):
        values = self.by_name.get(name, [])
        return values[0] if values else None

    def expand_sources(self, name):
        if name == "character":
            return []
        return list(self.by_name.get(name, []))

    def has_prop(self, obj, prop):
        return prop in obj.properties

    def state(self, obj, state):
        return state in self.states.get(obj.key, set())

    def add(self, action, *objs):
        params = []
        for obj in objs:
            if obj is None or obj.name == "character":
                continue
            params.extend([obj.name, obj.object_id])
        self.actions.append((action, params))

    def standup_if_needed(self):
        if "SITTING" in self.character_states or "LYING" in self.character_states:
            self.add("STANDUP")
            self.character_states.discard("SITTING")
            self.character_states.discard("LYING")

    def walk(self, obj):
        if obj is None or obj.name == "character":
            return
        self.standup_if_needed()
        if obj.key not in self.near:
            self.add("WALK", obj)
            self.near.add(obj.key)

    def open_if_needed(self, obj):
        if obj is None or not self.has_prop(obj, "CAN_OPEN"):
            return
        if self.state(obj, "OPEN"):
            return
        self.walk(obj)
        self.add("OPEN", obj)
        self.states[obj.key].discard("CLOSED")
        self.states[obj.key].add("OPEN")

    def close_if_needed(self, obj):
        if obj is None or not self.has_prop(obj, "CAN_OPEN"):
            return
        if self.state(obj, "CLOSED"):
            return
        self.walk(obj)
        self.add("CLOSE", obj)
        self.states[obj.key].discard("OPEN")
        self.states[obj.key].add("CLOSED")

    def make_accessible(self, obj):
        loc = self.locations.get(obj.key)
        if loc and loc[0] == "INSIDE":
            container = self.obj_from_key(loc[1])
            self.open_if_needed(container)

    def reach_target(self, obj):
        loc = self.locations.get(obj.key)
        if loc and loc[0] in {"INSIDE", "ON"}:
            container = self.obj_from_key(loc[1])
            if container:
                return container
        return obj

    def obj_from_key(self, key):
        name, object_id = key
        return self.object_by_name_id(name, object_id)

    def grab(self, obj):
        if obj is None or obj.key in self.held:
            return
        self.make_accessible(obj)
        self.walk(self.reach_target(obj))
        self.add("GRAB", obj)
        self.held.add(obj.key)

    def put_on(self, obj, target):
        if obj is None or target is None:
            return
        if self.locations.get(obj.key) == ("ON", target.key):
            return
        self.grab(obj)
        self.walk(target)
        self.add("PUTBACK", obj, target)
        self.held.discard(obj.key)
        self.locations[obj.key] = ("ON", target.key)

    def put_inside(self, obj, target):
        if obj is None or target is None:
            return
        if self.locations.get(obj.key) == ("INSIDE", target.key):
            return
        self.grab(obj)
        self.open_if_needed(target)
        self.walk(target)
        self.add("PUTIN", obj, target)
        self.held.discard(obj.key)
        self.locations[obj.key] = ("INSIDE", target.key)

    def switch_on(self, obj):
        if obj is None or self.state(obj, "ON"):
            return
        self.walk(obj)
        if self.has_prop(obj, "HAS_PLUG") and not self.state(obj, "PLUGGED_IN"):
            self.add("PLUGIN", obj)
            self.states[obj.key].discard("PLUGGED_OUT")
            self.states[obj.key].add("PLUGGED_IN")
        self.add("SWITCHON", obj)
        self.states[obj.key].discard("OFF")
        self.states[obj.key].add("ON")

    def switch_off(self, obj):
        if obj is None or self.state(obj, "OFF"):
            return
        self.walk(obj)
        self.add("SWITCHOFF", obj)
        self.states[obj.key].discard("ON")
        self.states[obj.key].add("OFF")

    def sit_or_lie(self, target, prefer_lie=False):
        self.walk(target)
        if prefer_lie or (target and self.has_prop(target, "LIEABLE") and not self.has_prop(target, "SITTABLE")):
            self.add("LIE", target)
            self.character_states.add("LYING")
        else:
            self.add("SIT", target)
            self.character_states.add("SITTING")

    def choose_for_action(self, action):
        action = action.upper()
        if action in {"SWITCHON", "SWITCHOFF"}:
            wanted = "ON" if action == "SWITCHON" else "OFF"
            for name, state in self.node_goals:
                if state == wanted:
                    obj = self.first(name)
                    if obj:
                        return [obj]
        if action in {"LOOKAT", "WATCH", "TURNTO", "TOUCH", "POINTAT"}:
            for src, rel, dst in self.edge_goals:
                if src == "character" and rel in {"FACING", "CLOSE"}:
                    obj = self.first(dst)
                    if obj:
                        return [obj]
        if action == "GRAB":
            for src, rel, dst in self.edge_goals:
                if src == "character" and rel in {"HOLDS_RH", "HOLDS_LH"}:
                    obj = self.first(dst)
                    if obj:
                        return [obj]
            for src, rel, _dst in self.edge_goals:
                if src != "character" and rel in {"ON", "INSIDE"}:
                    obj = self.first(src)
                    if obj:
                        return [obj]
        if action in {"READ", "TYPE"}:
            prop = "READABLE" if action == "READ" else "HAS_SWITCH"
            for obj in self.objects:
                if self.has_prop(obj, prop):
                    return [obj]
        if action in {"DRINK", "EAT", "CUT"}:
            props = {
                "DRINK": {"DRINKABLE", "RECIPIENT"},
                "EAT": {"EATABLE"},
                "CUT": {"EATABLE", "CUTABLE"},
            }[action]
            for obj in self.objects:
                if props.intersection(obj.properties):
                    return [obj]
        if action in {"WASH", "RINSE", "SCRUB", "WIPE", "SQUEEZE"}:
            for name, state in self.node_goals:
                if state == "CLEAN":
                    obj = self.first(name)
                    if obj:
                        return [obj]
            for obj in self.objects:
                if action == "SQUEEZE" and self.has_prop(obj, "CLOTHES"):
                    return [obj]
            for obj in self.objects:
                if obj.name != "character":
                    return [obj]
        if action == "POUR":
            src = next(
                (obj for obj in self.objects if {"POURABLE", "DRINKABLE"}.intersection(obj.properties)),
                None,
            )
            dst = next((obj for obj in self.objects if self.has_prop(obj, "RECIPIENT")), None)
            if src and dst:
                return [src, dst]
        if action in {"SIT", "LIE"}:
            prop = "SITTABLE" if action == "SIT" else "LIEABLE"
            for obj in self.objects:
                if self.has_prop(obj, prop):
                    return [obj]
        if action in {"PUSH", "PULL", "MOVE"}:
            for obj in self.objects:
                if self.has_prop(obj, "MOVABLE"):
                    return [obj]
        if action in {"OPEN", "CLOSE"}:
            state = "OPEN" if action == "OPEN" else "CLOSED"
            for name, goal_state in self.node_goals:
                if goal_state == state:
                    obj = self.first(name)
                    if obj:
                        return [obj]
            for obj in self.objects:
                if self.has_prop(obj, "CAN_OPEN"):
                    return [obj]
        return []

    def emit_action_goal(self, alternatives):
        preferred = None
        for action in alternatives:
            candidate = action.upper()
            if self.choose_for_action(candidate) or candidate in {"SLEEP", "WAKEUP", "STANDUP"}:
                preferred = candidate
                break
        if preferred is None:
            preferred = alternatives[0].upper()

        if preferred == "STANDUP":
            self.add("STANDUP")
            return
        if preferred in {"SLEEP", "WAKEUP"}:
            self.add(preferred)
            return

        objs = self.choose_for_action(preferred)
        if not objs:
            return
        if preferred == "GRAB":
            self.grab(objs[0])
            return
        if preferred == "OPEN":
            self.open_if_needed(objs[0])
            return
        if preferred == "CLOSE":
            self.close_if_needed(objs[0])
            return
        if preferred == "SWITCHON":
            self.switch_on(objs[0])
            return
        if preferred == "SWITCHOFF":
            self.switch_off(objs[0])
            return
        if preferred in {"SIT", "LIE"}:
            self.sit_or_lie(objs[0], prefer_lie=preferred == "LIE")
            return
        for obj in objs:
            self.walk(obj)
        self.add(preferred, *objs)

    def compile_edge_goals(self):
        for src, rel, dst in self.edge_goals:
            target = self.first(dst)
            if src == "character":
                if rel in {"CLOSE", "NEAR"}:
                    self.walk(target)
                elif rel == "FACING":
                    self.walk(target)
                    self.add("TURNTO", target)
                elif rel in {"HOLDS_RH", "HOLDS_LH"}:
                    self.grab(target)
                elif rel == "INSIDE":
                    self.walk(target)
                elif rel == "ON":
                    prefer_lie = any(
                        name == "character" and state == "LYING"
                        for name, state in self.node_goals
                    )
                    self.sit_or_lie(target, prefer_lie=prefer_lie)
                continue

            for obj in self.expand_sources(src):
                if rel == "ON" and dst == "character":
                    self.grab(obj)
                    self.add("PUTON", obj)
                    self.held.discard(obj.key)
                elif rel == "ON":
                    self.put_on(obj, target)
                elif rel == "INSIDE":
                    self.put_inside(obj, target)

    def compile_node_goals(self):
        final_closed = []
        for name, state in self.node_goals:
            if name == "character":
                if state == "SITTING":
                    sit_target = next(
                        (self.first(dst) for src, rel, dst in self.edge_goals if src == "character" and rel == "ON"),
                        None,
                    )
                    if sit_target:
                        self.sit_or_lie(sit_target, prefer_lie=False)
                elif state == "LYING":
                    lie_target = next(
                        (self.first(dst) for src, rel, dst in self.edge_goals if src == "character" and rel == "ON"),
                        None,
                    )
                    if lie_target:
                        self.sit_or_lie(lie_target, prefer_lie=True)
                continue

            obj = self.first(name)
            if obj is None:
                continue
            if state == "PLUGGED_IN":
                if self.has_prop(obj, "HAS_PLUG") and not self.state(obj, "PLUGGED_IN"):
                    self.walk(obj)
                    self.add("PLUGIN", obj)
                    self.states[obj.key].discard("PLUGGED_OUT")
                    self.states[obj.key].add("PLUGGED_IN")
            elif state == "PLUGGED_OUT":
                if self.has_prop(obj, "HAS_PLUG") and not self.state(obj, "PLUGGED_OUT"):
                    self.walk(obj)
                    self.add("PLUGOUT", obj)
                    self.states[obj.key].discard("PLUGGED_IN")
                    self.states[obj.key].add("PLUGGED_OUT")
            elif state == "ON":
                self.switch_on(obj)
            elif state == "OFF":
                self.switch_off(obj)
            elif state == "OPEN":
                self.open_if_needed(obj)
            elif state == "CLOSED":
                final_closed.append(obj)
            elif state == "CLEAN":
                if not self.state(obj, "CLEAN"):
                    self.walk(obj)
                    self.add("WIPE", obj)
                    self.states[obj.key].add("CLEAN")

        for obj in final_closed:
            self.close_if_needed(obj)

    def compile(self):
        for alternatives in self.action_goals:
            self.emit_action_goal(alternatives)
        self.compile_edge_goals()
        self.compile_node_goals()
        if not self.actions:
            first_non_character = next((obj for obj in self.objects if obj.name != "character"), None)
            if first_non_character:
                self.add("WALK", first_non_character)
        return self.actions


def render_actions(actions):
    lines = ["{"]
    for idx, (action, params) in enumerate(actions):
        suffix = "," if idx < len(actions) - 1 else ""
        lines.append(f'  "{action}": {json.dumps(params, ensure_ascii=False)}{suffix}')
    lines.append("}")
    return "\n".join(lines)


def generate_outputs(prompt_path):
    prompts = load_json(prompt_path)
    outputs = []
    stats = {
        "generated": 0,
        "empty": 0,
        "avg_actions": 0.0,
        "max_actions": 0,
    }
    total_actions = 0

    for item in prompts:
        parsed = parse_prompt(item["llm_prompt"])
        actions = VirtualHomeCompiler(parsed).compile()
        if not actions:
            stats["empty"] += 1
        total_actions += len(actions)
        stats["max_actions"] = max(stats["max_actions"], len(actions))
        stats["generated"] += 1
        outputs.append(
            {
                "identifier": item["identifier"],
                "llm_output": render_actions(actions),
            }
        )

    stats["avg_actions"] = round(total_actions / max(1, stats["generated"]), 2)
    return outputs, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompts",
        default="llm_prompts/virtualhome_action_sequencing_prompts.json",
    )
    parser.add_argument("--base-dir", default="best")
    parser.add_argument("--output-dir", default="best_vhas_2stage")
    parser.add_argument("--outputs-only", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    prompt_path = root / args.prompts
    output_dir = root / args.output_dir

    outputs, stats = generate_outputs(prompt_path)

    if not args.outputs_only:
        base_dir = root / args.base_dir
        if output_dir.exists():
            shutil.rmtree(output_dir)
        shutil.copytree(base_dir, output_dir)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

    out_path = output_dir / "virtualhome_action_sequencing_outputs.json"
    out_path.write_text(json.dumps(outputs, ensure_ascii=False, indent=4), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), **stats}, indent=2))


if __name__ == "__main__":
    main()
