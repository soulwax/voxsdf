import 'dart:math' as math;

import 'field.dart';
import 'primitives.dart';

/// Indexed triangle mesh extracted from an [SdfField] isosurface via dual Surface Nets.
final class SdfMeshResult {
  final List<SdfVec3> positions;
  final List<SdfVec3> normals;
  final List<int> indices;

  const SdfMeshResult({
    required this.positions,
    required this.normals,
    required this.indices,
  });

  int get vertexCount => positions.length;
  int get triangleCount => indices.length ~/ 3;
}

/// Smooth dual Surface Nets isosurface extractor generating indexed vertices & gradient normals.
final class SdfSurfaceNets {
  static const List<(int, int)> _cubeEdges = [
    (0, 1), (2, 3), (4, 5), (6, 7),
    (0, 2), (1, 3), (4, 6), (5, 7),
    (0, 4), (1, 5), (2, 6), (3, 7),
  ];

  static const List<(int, int, int)> _cornerOffsets = [
    (0, 0, 0),
    (1, 0, 0),
    (0, 1, 0),
    (1, 1, 0),
    (0, 0, 1),
    (1, 0, 1),
    (0, 1, 1),
    (1, 1, 1),
  ];

  static SdfMeshResult extract(
    SdfField field, {
    double isoLevel = 0.0,
  }) {
    final cxMax = field.nx - 1;
    final cyMax = field.ny - 1;
    final czMax = field.nz - 1;
    if (cxMax <= 1 || cyMax <= 1 || czMax <= 1) {
      return const SdfMeshResult(positions: [], normals: [], indices: []);
    }

    final cellVertexIndices = List<int>.filled(cxMax * cyMax * czMax, -1);
    int cellIdx(int x, int y, int z) => (z * cyMax + y) * cxMax + x;

    final positions = <SdfVec3>[];
    final normals = <SdfVec3>[];
    final cornerD = List<double>.filled(8, 0.0);

    // 1. Place one dual vertex per surface-crossing cell at average of zero-crossing edge points
    for (var z = 0; z < czMax; z++) {
      for (var y = 0; y < cyMax; y++) {
        for (var x = 0; x < cxMax; x++) {
          var insideMask = 0;
          for (var c = 0; c < 8; c++) {
            final off = _cornerOffsets[c];
            final d = field.atGrid(x + off.$1, y + off.$2, z + off.$3) - isoLevel;
            cornerD[c] = d;
            if (d < 0.0) insideMask |= (1 << c);
          }
          if (insideMask == 0 || insideMask == 255) continue;

          var sumX = 0.0;
          var sumY = 0.0;
          var sumZ = 0.0;
          var crossings = 0;

          for (final e in _cubeEdges) {
            final d0 = cornerD[e.$1];
            final d1 = cornerD[e.$2];
            if ((d0 < 0.0) == (d1 < 0.0)) continue;
            final denom = d0 - d1;
            final t = denom.abs() < 1e-9 ? 0.5 : (d0 / denom).clamp(0.0, 1.0).toDouble();
            final o0 = _cornerOffsets[e.$1];
            final o1 = _cornerOffsets[e.$2];
            sumX += o0.$1 + (o1.$1 - o0.$1) * t;
            sumY += o0.$2 + (o1.$2 - o0.$2) * t;
            sumZ += o0.$3 + (o1.$3 - o0.$3) * t;
            crossings++;
          }

          if (crossings == 0) continue;
          final inv = 1.0 / crossings;
          final worldPos = field.gridToWorld(
            x + sumX * inv,
            y + sumY * inv,
            z + sumZ * inv,
          );
          final vIdx = positions.length;
          positions.add(worldPos);
          normals.add(field.computeGradientNormal(worldPos));
          cellVertexIndices[cellIdx(x, y, z)] = vIdx;
        }
      }
    }

    // 2. Stitch quads (as 2 triangles) across every interior edge crossing the isosurface
    final indices = <int>[];

    void emitQuad(int a, int b, int c, int d, bool flip) {
      if (a < 0 || b < 0 || c < 0 || d < 0) return;
      if (!flip) {
        indices.addAll([a, b, c, a, c, d]);
      } else {
        indices.addAll([a, c, b, a, d, c]);
      }
    }

    // X-edges: shared by 4 cells around y-1..y, z-1..z
    for (var z = 1; z < czMax; z++) {
      for (var y = 1; y < cyMax; y++) {
        for (var x = 0; x < cxMax; x++) {
          final insideA = (field.atGrid(x, y, z) - isoLevel) < 0.0;
          final insideB = (field.atGrid(x + 1, y, z) - isoLevel) < 0.0;
          if (insideA == insideB) continue;
          emitQuad(
            cellVertexIndices[cellIdx(x, y - 1, z - 1)],
            cellVertexIndices[cellIdx(x, y, z - 1)],
            cellVertexIndices[cellIdx(x, y, z)],
            cellVertexIndices[cellIdx(x, y - 1, z)],
            !insideA,
          );
        }
      }
    }

    // Y-edges: shared by 4 cells around x-1..x, z-1..z
    for (var z = 1; z < czMax; z++) {
      for (var y = 0; y < cyMax; y++) {
        for (var x = 1; x < cxMax; x++) {
          final insideA = (field.atGrid(x, y, z) - isoLevel) < 0.0;
          final insideB = (field.atGrid(x, y + 1, z) - isoLevel) < 0.0;
          if (insideA == insideB) continue;
          emitQuad(
            cellVertexIndices[cellIdx(x - 1, y, z - 1)],
            cellVertexIndices[cellIdx(x - 1, y, z)],
            cellVertexIndices[cellIdx(x, y, z)],
            cellVertexIndices[cellIdx(x, y, z - 1)],
            !insideA,
          );
        }
      }
    }

    // Z-edges: shared by 4 cells around x-1..x, y-1..y
    for (var z = 0; z < czMax; z++) {
      for (var y = 1; y < cyMax; y++) {
        for (var x = 1; x < cxMax; x++) {
          final insideA = (field.atGrid(x, y, z) - isoLevel) < 0.0;
          final insideB = (field.atGrid(x, y, z + 1) - isoLevel) < 0.0;
          if (insideA == insideB) continue;
          emitQuad(
            cellVertexIndices[cellIdx(x - 1, y - 1, z)],
            cellVertexIndices[cellIdx(x, y - 1, z)],
            cellVertexIndices[cellIdx(x, y, z)],
            cellVertexIndices[cellIdx(x - 1, y, z)],
            !insideA,
          );
        }
      }
    }

    // Remove unused edge cases? Wait: every emitted index is in 0..positions.length-1; ensure unit length normals.
    for (var i = 0; i < normals.length; i++) {
      final n = normals[i];
      if (n.lengthSquared < 1e-8) {
        normals[i] = const SdfVec3(0.0, 1.0, 0.0);
      } else {
        normals[i] = n.normalized;
      }
    }

    return SdfMeshResult(
      positions: positions,
      normals: normals,
      indices: indices,
    );
  }
}

/// Optional bounding summary helper for extruded Surface Nets meshes.
({SdfVec3 min, SdfVec3 max}) computeSdfMeshBounds(SdfMeshResult mesh) {
  if (mesh.positions.isEmpty) {
    return (min: SdfVec3.zero, max: SdfVec3.zero);
  }
  var minX = mesh.positions.first.x;
  var minY = mesh.positions.first.y;
  var minZ = mesh.positions.first.z;
  var maxX = minX;
  var maxY = minY;
  var maxZ = minZ;
  for (var i = 1; i < mesh.positions.length; i++) {
    final p = mesh.positions[i];
    minX = math.min(minX, p.x);
    minY = math.min(minY, p.y);
    minZ = math.min(minZ, p.z);
    maxX = math.max(maxX, p.x);
    maxY = math.max(maxY, p.y);
    maxZ = math.max(maxZ, p.z);
  }
  return (min: SdfVec3(minX, minY, minZ), max: SdfVec3(maxX, maxY, maxZ));
}
