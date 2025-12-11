from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - torch is always available in runtime but keep guard for linting
    torch = None  # type: ignore


@dataclass
class Node:
    """Simple container storing a node id and its coordinates."""

    id: int
    coords: Tuple[float, float, float]


@dataclass
class ElementConnectivity:
    """Stores element id, type (C3D8, etc.) and node ids."""

    id: int
    type: str
    nodes: List[int]


@dataclass
class NodeSet:
    name: str
    nodes: List[int]


@dataclass
class ElementSet:
    name: str
    elements: List[int]


@dataclass
class Surface:
    name: str
    faces: List[Tuple[int, int]]


class Mesh:
    """Python port of FEM-ML's mesh data structure."""

    def __init__(self) -> None:
        self.nodes: Dict[int, Node] = {}
        self.elements: Dict[int, ElementConnectivity] = {}
        self.node_sets: Dict[str, NodeSet] = {}
        self.element_sets: Dict[str, ElementSet] = {}
        self.surfaces: Dict[str, Surface] = {}
        self._node_index_map: Optional[Dict[int, int]] = None
        self._element_index_map: Optional[Dict[int, int]] = None

    # --- entity creation -------------------------------------------------
    def add_node(self, node: Node) -> None:
        self.nodes[node.id] = node
        self._node_index_map = None

    def add_element(self, element: ElementConnectivity) -> None:
        self.elements[element.id] = element
        self._element_index_map = None

    def add_node_set(self, name: str, nodes: Iterable[int]) -> None:
        self.node_sets[name] = NodeSet(name=name, nodes=list(nodes))

    def add_element_set(self, name: str, elements: Iterable[int]) -> None:
        self.element_sets[name] = ElementSet(name=name, elements=list(elements))

    def add_surface(self, name: str, faces: Iterable[Tuple[int, int]]) -> None:
        self.surfaces[name] = Surface(name=name, faces=list(faces))

    # --- queries ---------------------------------------------------------
    def summary(self) -> Dict[str, int]:
        """Return counts for UI display."""
        return {
            "nodes": len(self.nodes),
            "elements": len(self.elements),
            "node_sets": len(self.node_sets),
            "element_sets": len(self.element_sets),
            "surfaces": len(self.surfaces),
        }

    def bounding_box(self) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        coords = np.array([node.coords for node in self.nodes.values()], dtype=float)
        if coords.size == 0:
            return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
        mins = tuple(coords.min(axis=0).tolist())
        maxs = tuple(coords.max(axis=0).tolist())
        return mins, maxs

    def _ensure_index_maps(self) -> None:
        if self._node_index_map is not None and self._element_index_map is not None:
            return
        node_ids = sorted(self.nodes.keys())
        self._node_index_map = {nid: idx for idx, nid in enumerate(node_ids)}
        element_ids = sorted(self.elements.keys())
        self._element_index_map = {eid: idx for idx, eid in enumerate(element_ids)}

    def node_indices_from_set(self, name: str) -> List[int]:
        self._ensure_index_maps()
        if name not in self.node_sets or self._node_index_map is None:
            return []
        result = []
        for nid in self.node_sets[name].nodes:
            if nid in self._node_index_map:
                result.append(self._node_index_map[nid])
        return result

    def node_ids(self) -> List[int]:
        return sorted(self.nodes.keys())

    def element_ids(self) -> List[int]:
        return sorted(self.elements.keys())

    # --- conversions -----------------------------------------------------
    def to_numpy(self) -> Tuple[np.ndarray, np.ndarray, List[int], List[int]]:
        """Return (coords, connectivity, node_ids, element_ids)."""
        self._ensure_index_maps()
        node_ids = self.node_ids()
        coords = np.array([self.nodes[nid].coords for nid in node_ids], dtype=float)
        element_ids = self.element_ids()
        connectivity: List[List[int]] = []
        if self._node_index_map is None:
            raise RuntimeError("Node index map not built")
        for eid in element_ids:
            elem = self.elements[eid]
            connectivity.append([self._node_index_map[nid] for nid in elem.nodes])
        return coords, np.array(connectivity, dtype=int), node_ids, element_ids

    def to_torch(self, dtype: Optional["torch.dtype"] = None, device: Optional["torch.device"] = None):
        if torch is None:
            raise ImportError("torch is required to convert mesh to tensors")
        coords, conn, _, _ = self.to_numpy()
        coords_t = torch.tensor(coords, dtype=dtype or torch.float64, device=device)
        conn_t = torch.tensor(conn, dtype=torch.long, device=device)
        return coords_t, conn_t


class AbaqusInpImporter:
    """Parses Abaqus .inp meshes. Ported from FEM-ML C++ implementation."""

    def __init__(self, verbose: bool = True) -> None:
        self.verbose = verbose
        self._pending_line: Optional[str] = None

    # --- public API ------------------------------------------------------
    def import_file(self, path: str) -> Mesh:
        mesh = Mesh()
        with open(path, "r", encoding="utf-8") as handle:
            if self.verbose:
                print(f"[AbaqusImporter] Importing {path}")
            while True:
                line = self._next_line(handle)
                if line is None:
                    break
                upper = line.upper()
                if upper.startswith("*NODE"):
                    self._parse_nodes(handle, mesh)
                elif upper.startswith("*ELEMENT"):
                    elem_type = self._extract_option(upper, "TYPE") or "C3D8"
                    self._parse_elements(handle, mesh, elem_type)
                elif upper.startswith("*NSET"):
                    name = self._extract_option(upper, "NSET") or "SET"
                    generate = "GENERATE" in upper
                    self._parse_node_set(handle, mesh, name, generate)
                elif upper.startswith("*ELSET"):
                    name = self._extract_option(upper, "ELSET") or "ELSET"
                    generate = "GENERATE" in upper
                    self._parse_element_set(handle, mesh, name, generate)
                elif upper.startswith("*SURFACE"):
                    name = self._extract_option(upper, "NAME") or "SURFACE"
                    self._parse_surface(handle, mesh, name)
                else:
                    # ignore other sections
                    continue
        if self.verbose:
            mins, maxs = mesh.bounding_box()
            summary = mesh.summary()
            print(
                f"[AbaqusImporter] nodes={summary['nodes']} elements={summary['elements']} "
                f"bbox_min={mins} bbox_max={maxs}"
            )
        return mesh

    # --- parsing helpers -------------------------------------------------
    def _parse_nodes(self, handle, mesh: Mesh) -> None:
        count = 0
        for raw in self._collect_block(handle):
            parts = [p.strip() for p in raw.split(",") if p.strip()]
            if len(parts) < 4:
                continue
            node_id = int(parts[0])
            x, y, z = (self._to_float(parts[1]), self._to_float(parts[2]), self._to_float(parts[3]))
            mesh.add_node(Node(id=node_id, coords=(x, y, z)))
            count += 1
        if self.verbose:
            print(f"  [+] Parsed {count} nodes")

    def _parse_elements(self, handle, mesh: Mesh, elem_type: str) -> None:
        count = 0
        for raw in self._collect_block(handle):
            parts = [p.strip() for p in raw.split(",") if p.strip()]
            if len(parts) < 2:
                continue
            elem_id = int(parts[0])
            nodes = [int(p) for p in parts[1:]]
            mesh.add_element(ElementConnectivity(id=elem_id, type=elem_type, nodes=nodes))
            count += 1
        if self.verbose:
            print(f"  [+] Parsed {count} {elem_type} elements")

    def _parse_node_set(self, handle, mesh: Mesh, name: str, generate: bool) -> None:
        nodes: List[int] = []
        for raw in self._collect_block(handle):
            parts = [p.strip() for p in raw.split(",") if p.strip()]
            if not parts:
                continue
            if generate and len(parts) >= 2:
                start = int(parts[0])
                end = int(parts[1])
                step = int(parts[2]) if len(parts) >= 3 else 1
                nodes.extend(range(start, end + step, step))
            else:
                nodes.extend(int(p) for p in parts)
        mesh.add_node_set(name, nodes)
        if self.verbose:
            print(f"  [+] Parsed node set '{name}' ({len(nodes)} nodes)")

    def _parse_element_set(self, handle, mesh: Mesh, name: str, generate: bool) -> None:
        elements: List[int] = []
        for raw in self._collect_block(handle):
            parts = [p.strip() for p in raw.split(",") if p.strip()]
            if not parts:
                continue
            if generate and len(parts) >= 2:
                start = int(parts[0])
                end = int(parts[1])
                step = int(parts[2]) if len(parts) >= 3 else 1
                elements.extend(range(start, end + step, step))
            else:
                elements.extend(int(p) for p in parts)
        mesh.add_element_set(name, elements)
        if self.verbose:
            print(f"  [+] Parsed element set '{name}' ({len(elements)} elements)")

    def _parse_surface(self, handle, mesh: Mesh, name: str) -> None:
        faces: List[Tuple[int, int]] = []
        for raw in self._collect_block(handle):
            parts = [p.strip() for p in raw.split(",") if p.strip()]
            if len(parts) < 2:
                continue
            elem_id = int(parts[0])
            face_entry = parts[1]
            if face_entry.upper().startswith("S"):
                face_id = int(face_entry[1:])
            else:
                face_id = int(face_entry)
            faces.append((elem_id, face_id))
        mesh.add_surface(name, faces)
        if self.verbose:
            print(f"  [+] Parsed surface '{name}' ({len(faces)} faces)")

    def _collect_block(self, handle) -> List[str]:
        block: List[str] = []
        while True:
            line = self._next_line(handle)
            if line is None:
                break
            if line.startswith("*"):
                self._pending_line = line
                break
            block.append(line)
        return block

    def _next_line(self, handle) -> Optional[str]:
        if self._pending_line is not None:
            line = self._pending_line
            self._pending_line = None
            return line
        for raw in handle:
            stripped = raw.strip()
            if not stripped:
                continue
            if stripped.startswith("**"):
                continue
            return stripped
        return None

    @staticmethod
    def _extract_option(line: str, key: str) -> Optional[str]:
        key_upper = f"{key.upper()}="
        idx = line.find(key_upper)
        if idx == -1:
            return None
        start = idx + len(key_upper)
        end = line.find(",", start)
        if end == -1:
            end = len(line)
        return line[start:end].strip()

    @staticmethod
    def _to_float(entry: str) -> float:
        return float(entry.replace("D", "E").replace("d", "E"))

