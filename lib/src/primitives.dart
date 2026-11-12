import 'dart:math' as math;

/// Lightweight 3D vector for zero-dependency SDF field sampling and Surface Nets extraction.
final class SdfVec3 {
  final double x;
  final double y;
  final double z;

  const SdfVec3(this.x, this.y, this.z);

  static const SdfVec3 zero = SdfVec3(0.0, 0.0, 0.0);

  SdfVec3 operator +(SdfVec3 other) => SdfVec3(x + other.x, y + other.y, z + other.z);
  SdfVec3 operator -(SdfVec3 other) => SdfVec3(x - other.x, y - other.y, z - other.z);
  SdfVec3 operator *(double s) => SdfVec3(x * s, y * s, z * s);
  SdfVec3 operator -() => SdfVec3(-x, -y, -z);

  double dot(SdfVec3 o) => x * o.x + y * o.y + z * o.z;

  SdfVec3 cross(SdfVec3 o) => SdfVec3(
        y * o.z - z * o.y,
        z * o.x - x * o.z,
        x * o.y - y * o.x,
      );

  double get lengthSquared => x * x + y * y + z * z;
  double get length => math.sqrt(lengthSquared);

  SdfVec3 get normalized {
    final len = length;
    if (len < 1e-12) return const SdfVec3(0.0, 1.0, 0.0);
    return SdfVec3(x / len, y / len, z / len);
  }

  SdfVec3 abs() => SdfVec3(x.abs(), y.abs(), z.abs());
}

/// CSG boolean combination operator for SDF sculpting brushes.
enum SdfOp {
  union,
  smoothUnion,
  subtract,
  intersect,
}

/// Base analytic 3D signed distance function primitive.
sealed class SdfPrimitive {
  const SdfPrimitive();

  /// Returns exact signed Euclidean distance (< 0 inside surface, > 0 outside).
  double sample(SdfVec3 p);
}

final class SdfSphere extends SdfPrimitive {
  final SdfVec3 center;
  final double radius;

  const SdfSphere({
    this.center = SdfVec3.zero,
    this.radius = 1.0,
  });

  @override
  double sample(SdfVec3 p) => (p - center).length - radius;
}

final class SdfBox extends SdfPrimitive {
  final SdfVec3 center;
  final SdfVec3 halfExtents;
  final double rounding;

  const SdfBox({
    this.center = SdfVec3.zero,
    this.halfExtents = const SdfVec3(1.0, 1.0, 1.0),
    this.rounding = 0.0,
  });

  @override
  double sample(SdfVec3 p) {
    final q = (p - center).abs() - halfExtents;
    final outX = math.max(q.x, 0.0);
    final outY = math.max(q.y, 0.0);
    final outZ = math.max(q.z, 0.0);
    final outside = math.sqrt(outX * outX + outY * outY + outZ * outZ);
    final inside = math.min(math.max(q.x, math.max(q.y, q.z)), 0.0);
    return outside + inside - rounding;
  }
}

final class SdfCapsule extends SdfPrimitive {
  final SdfVec3 a;
  final SdfVec3 b;
  final double radius;

  const SdfCapsule({
    required this.a,
    required this.b,
    this.radius = 0.5,
  });

  @override
  double sample(SdfVec3 p) {
    final pa = p - a;
    final ba = b - a;
    final lenSq = math.max(1e-9, ba.lengthSquared);
    final h = (pa.dot(ba) / lenSq).clamp(0.0, 1.0).toDouble();
    return (pa - ba * h).length - radius;
  }
}

final class SdfTorus extends SdfPrimitive {
  final SdfVec3 center;
  final double majorRadius;
  final double minorRadius;

  const SdfTorus({
    this.center = SdfVec3.zero,
    this.majorRadius = 1.2,
    this.minorRadius = 0.35,
  });

  @override
  double sample(SdfVec3 p) {
    final q = p - center;
    final qx = math.sqrt(q.x * q.x + q.z * q.z) - majorRadius;
    return math.sqrt(qx * qx + q.y * q.y) - minorRadius;
  }
}

/// Sculpting brush pairing an [SdfPrimitive] with a CSG boolean operation.
final class SdfBrush {
  final SdfPrimitive primitive;
  final SdfOp op;
  final double smoothness;

  const SdfBrush({
    required this.primitive,
    this.op = SdfOp.union,
    this.smoothness = 0.25,
  });

  static double smoothMin(double a, double b, double k) {
    if (k <= 1e-6) return math.min(a, b);
    final h = (0.5 + 0.5 * (b - a) / k).clamp(0.0, 1.0).toDouble();
    return b * (1.0 - h) + a * h - k * h * (1.0 - h);
  }

  /// Combines [currentDistance] with [primitive.sample(p)] using [op].
  double evaluate(double currentDistance, SdfVec3 p) {
    final d = primitive.sample(p);
    switch (op) {
      case SdfOp.union:
        return math.min(currentDistance, d);
      case SdfOp.smoothUnion:
        return smoothMin(currentDistance, d, smoothness);
      case SdfOp.subtract:
        return math.max(currentDistance, -d);
      case SdfOp.intersect:
        return math.max(currentDistance, d);
    }
  }
}
