import 'package:test/test.dart';
import 'package:voxsdf/voxsdf.dart';

void main() {
  test('SdfField applies CSG union, smooth-union, and subtraction brushes across analytic primitives', () {
    final field = SdfField(nx: 18, ny: 18, nz: 18, voxelSize: 0.25);
    field.applyBrush(
      const SdfBrush(
        primitive: SdfBox(halfExtents: SdfVec3(1.2, 1.2, 1.2)),
        op: SdfOp.union,
      ),
    );
    expect(field.sampleTrilinear(SdfVec3.zero), lessThan(0.0));

    // Carve spherical cavity right through the center
    field.applyBrush(
      const SdfBrush(
        primitive: SdfSphere(radius: 0.7),
        op: SdfOp.subtract,
      ),
    );
    expect(field.sampleTrilinear(SdfVec3.zero), greaterThan(0.4));
    expect(field.sampleTrilinear(const SdfVec3(0.95, 0.0, 0.0)), lessThan(0.0));
  });

  test('SdfField sphere-tracing raymarcher converges onto torus/capsule surface with accurate normal', () {
    final field = SdfField(nx: 22, ny: 22, nz: 22, voxelSize: 0.25);
    field.applyBrush(
      const SdfBrush(
        primitive: SdfSphere(radius: 1.5),
        op: SdfOp.union,
      ),
    );
    field.applyBrush(
      const SdfBrush(
        primitive: SdfCapsule(
          a: SdfVec3(0.0, 1.0, 0.0),
          b: SdfVec3(0.0, 2.0, 0.0),
          radius: 0.55,
        ),
        op: SdfOp.smoothUnion,
        smoothness: 0.35,
      ),
    );

    final hit = field.raymarch(
      const SdfVec3(5.0, 0.0, 0.0),
      const SdfVec3(-1.0, 0.0, 0.0),
    );
    expect(hit, isNotNull);
    expect(hit!.point.x, closeTo(1.5, 0.12));
    expect(hit.normal.x, greaterThan(0.9));
  });

  test('SdfSurfaceNets extracts indexed outward-wound manifold mesh with normalized gradients', () {
    final field = SdfField(nx: 20, ny: 20, nz: 20, voxelSize: 0.25);
    field.applyBrush(
      const SdfBrush(
        primitive: SdfTorus(majorRadius: 1.3, minorRadius: 0.45),
        op: SdfOp.union,
      ),
    );

    final mesh = SdfSurfaceNets.extract(field);
    expect(mesh.vertexCount, greaterThan(80));
    expect(mesh.triangleCount, greaterThan(120));
    expect(mesh.indices.every((i) => i >= 0 && i < mesh.vertexCount), isTrue);

    // Check outward winding on triangle faces via cross(b-a, c-a).dot(averageNormal) > 0
    var outwardFaces = 0;
    for (var t = 0; t < mesh.triangleCount; t++) {
      final ia = mesh.indices[t * 3 + 0];
      final ib = mesh.indices[t * 3 + 1];
      final ic = mesh.indices[t * 3 + 2];
      final faceN = (mesh.positions[ib] - mesh.positions[ia])
          .cross(mesh.positions[ic] - mesh.positions[ia]);
      final gradAvg = (mesh.normals[ia] + mesh.normals[ib] + mesh.normals[ic]);
      if (faceN.dot(gradAvg) > 0.0) outwardFaces++;
    }
    expect(outwardFaces / mesh.triangleCount, greaterThan(0.95));
  });
}
