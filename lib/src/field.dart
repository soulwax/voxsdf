import 'dart:math' as math;
import 'dart:typed_data';

import 'primitives.dart';

/// Closest-hit surface record returned by [SdfField.raymarch].
final class SdfRayHit {
  final double distance;
  final SdfVec3 point;
  final SdfVec3 normal;
  final int steps;

  const SdfRayHit({
    required this.distance,
    required this.point,
    required this.normal,
    required this.steps,
  });
}

/// Dense 3D voxel Signed Distance Field grid supporting CSG sculpting, trilinear lookup & sphere-tracing.
final class SdfField {
  final int nx;
  final int ny;
  final int nz;
  final double voxelSize;
  final SdfVec3 minCorner;
  final Float32List distances;

  SdfField({
    required this.nx,
    required this.ny,
    required this.nz,
    this.voxelSize = 0.25,
    SdfVec3? minCorner,
    double initialDistance = 32.0,
  })  : minCorner = minCorner ??
            SdfVec3(
              -(nx - 1) * voxelSize * 0.5,
              -(ny - 1) * voxelSize * 0.5,
              -(nz - 1) * voxelSize * 0.5,
            ),
        distances = Float32List(math.max(8, nx * ny * nz))..fillRange(0, math.max(8, nx * ny * nz), initialDistance);

  SdfVec3 get maxCorner => SdfVec3(
        minCorner.x + (nx - 1) * voxelSize,
        minCorner.y + (ny - 1) * voxelSize,
        minCorner.z + (nz - 1) * voxelSize,
      );

  int index(int ix, int iy, int iz) => (iz * ny + iy) * nx + ix;

  double atGrid(int ix, int iy, int iz) {
    final cx = ix.clamp(0, nx - 1);
    final cy = iy.clamp(0, ny - 1);
    final cz = iz.clamp(0, nz - 1);
    return distances[index(cx, cy, cz)];
  }

  void setAtGrid(int ix, int iy, int iz, double value) {
    if (ix < 0 || ix >= nx || iy < 0 || iy >= ny || iz < 0 || iz >= nz) return;
    distances[index(ix, iy, iz)] = value;
  }

  SdfVec3 gridToWorld(double gx, double gy, double gz) => SdfVec3(
        minCorner.x + gx * voxelSize,
        minCorner.y + gy * voxelSize,
        minCorner.z + gz * voxelSize,
      );

  void clear([double initialDistance = 32.0]) {
    distances.fillRange(0, distances.length, initialDistance);
  }

  /// Applies a CSG [SdfBrush] across every voxel sample in the volume.
  void applyBrush(SdfBrush brush) {
    for (var iz = 0; iz < nz; iz++) {
      final wz = minCorner.z + iz * voxelSize;
      for (var iy = 0; iy < ny; iy++) {
        final wy = minCorner.y + iy * voxelSize;
        for (var ix = 0; ix < nx; ix++) {
          final wx = minCorner.x + ix * voxelSize;
          final idx = index(ix, iy, iz);
          distances[idx] = brush.evaluate(distances[idx], SdfVec3(wx, wy, wz));
        }
      }
    }
  }

  /// Trilinearly interpolates signed distance inside bounds and extends positively outside AABB.
  double sampleTrilinear(SdfVec3 worldPos) {
    final gx = (worldPos.x - minCorner.x) / voxelSize;
    final gy = (worldPos.y - minCorner.y) / voxelSize;
    final gz = (worldPos.z - minCorner.z) / voxelSize;

    final maxGx = (nx - 1).toDouble();
    final maxGy = (ny - 1).toDouble();
    final maxGz = (nz - 1).toDouble();

    final outDx = gx < 0.0
        ? -gx * voxelSize
        : (gx > maxGx ? (gx - maxGx) * voxelSize : 0.0);
    final outDy = gy < 0.0
        ? -gy * voxelSize
        : (gy > maxGy ? (gy - maxGy) * voxelSize : 0.0);
    final outDz = gz < 0.0
        ? -gz * voxelSize
        : (gz > maxGz ? (gz - maxGz) * voxelSize : 0.0);
    final outBox = math.sqrt(outDx * outDx + outDy * outDy + outDz * outDz);

    final cgx = gx.clamp(0.0, maxGx).toDouble();
    final cgy = gy.clamp(0.0, maxGy).toDouble();
    final cgz = gz.clamp(0.0, maxGz).toDouble();

    final x0 = cgx.floor().clamp(0, nx - 2);
    final y0 = cgy.floor().clamp(0, ny - 2);
    final z0 = cgz.floor().clamp(0, nz - 2);
    final tx = (cgx - x0).clamp(0.0, 1.0).toDouble();
    final ty = (cgy - y0).clamp(0.0, 1.0).toDouble();
    final tz = (cgz - z0).clamp(0.0, 1.0).toDouble();

    final c000 = atGrid(x0, y0, z0);
    final c100 = atGrid(x0 + 1, y0, z0);
    final c010 = atGrid(x0, y0 + 1, z0);
    final c110 = atGrid(x0 + 1, y0 + 1, z0);
    final c001 = atGrid(x0, y0, z0 + 1);
    final c101 = atGrid(x0 + 1, y0, z0 + 1);
    final c011 = atGrid(x0, y0 + 1, z0 + 1);
    final c111 = atGrid(x0 + 1, y0 + 1, z0 + 1);

    final c00 = c000 * (1.0 - tx) + c100 * tx;
    final c10 = c010 * (1.0 - tx) + c110 * tx;
    final c01 = c001 * (1.0 - tx) + c101 * tx;
    final c11 = c011 * (1.0 - tx) + c111 * tx;

    final c0 = c00 * (1.0 - ty) + c10 * ty;
    final c1 = c01 * (1.0 - ty) + c11 * ty;
    final tri = c0 * (1.0 - tz) + c1 * tz;

    if (outBox > 0.0) {
      return outBox + math.max(0.002, tri);
    }
    return tri;
  }

  /// Estimates normalized surface outward gradient via central differences.
  SdfVec3 computeGradientNormal(SdfVec3 p, {double eps = 0.0}) {
    final h = eps > 0.0 ? eps : voxelSize * 0.5;
    final dx = sampleTrilinear(SdfVec3(p.x + h, p.y, p.z)) -
        sampleTrilinear(SdfVec3(p.x - h, p.y, p.z));
    final dy = sampleTrilinear(SdfVec3(p.x, p.y + h, p.z)) -
        sampleTrilinear(SdfVec3(p.x, p.y - h, p.z));
    final dz = sampleTrilinear(SdfVec3(p.x, p.y, p.z + h)) -
        sampleTrilinear(SdfVec3(p.x, p.y, p.z - h));
    return SdfVec3(dx, dy, dz).normalized;
  }

  /// Sphere-traces a ray from [origin] along [direction] returning the first isosurface crossing.
  SdfRayHit? raymarch(
    SdfVec3 origin,
    SdfVec3 direction, {
    double maxDistance = 64.0,
    int maxSteps = 128,
    double hitEpsilon = 2e-3,
    double isoLevel = 0.0,
  }) {
    final dir = direction.normalized;
    var t = 0.0;
    for (var step = 1; step <= maxSteps && t <= maxDistance; step++) {
      final pos = origin + dir * t;
      final d = sampleTrilinear(pos) - isoLevel;
      if (d <= hitEpsilon) {
        return SdfRayHit(
          distance: t,
          point: pos,
          normal: computeGradientNormal(pos),
          steps: step,
        );
      }
      t += math.max(d * 0.9, voxelSize * 0.05);
    }
    return null;
  }
}
