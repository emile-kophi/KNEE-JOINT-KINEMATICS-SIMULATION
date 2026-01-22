using System.Collections.Generic;
using UnityEngine;

[RequireComponent(typeof(MeshFilter))]
public class PhysicalSurfaceContactF_T : MonoBehaviour
{
    public MeshFilter femurMeshFilter;

    [Header("Reference frames")]
    public Transform femurPivot;
    public Transform tibiaPivot;

    [Header("Landmarks")]
    public Transform femurDistalLandmark;
    public Transform tibiaProximalLandmark;

    [Header("Contact settings")]
    public float contactThreshold = 0.001f; // 1 mm
    public float normalDotThreshold = -0.3f; // orientamento minimo verso femore
    public int samplingStep = 1;

    Mesh tibiaMesh;
    Vector3[] tibiaVertices;
    Vector3[] tibiaNormals;
    Color[] colors;

    Vector3[] femurVertices;
    int[] femurTriangles;

    List<int> proximalTibiaIndices = new List<int>();
    List<int> distalFemurTriangles = new List<int>();

    float femurDistalY;
    float tibiaProximalY;

    void Start()
    {
        tibiaMesh = GetComponent<MeshFilter>().mesh;
        tibiaVertices = tibiaMesh.vertices;
        tibiaNormals = tibiaMesh.normals;
        colors = new Color[tibiaVertices.Length];

        femurVertices = femurMeshFilter.mesh.vertices;
        femurTriangles = femurMeshFilter.mesh.triangles;

        femurDistalY = femurPivot
            .InverseTransformPoint(femurDistalLandmark.position).y;

        tibiaProximalY = tibiaPivot
            .InverseTransformPoint(tibiaProximalLandmark.position).y + 0.01f;

        PrecomputeRegions();
    }

    void PrecomputeRegions()
    {
        proximalTibiaIndices.Clear();
        distalFemurTriangles.Clear();

        // --- TIBIA PROSSIMALE (vertici target) ---
        for (int i = 0; i < tibiaVertices.Length; i++)
        {
            Vector3 vWorld = transform.TransformPoint(tibiaVertices[i]);
            Vector3 vInPivot = tibiaPivot.InverseTransformPoint(vWorld);

            if (vInPivot.y > tibiaProximalY)
                proximalTibiaIndices.Add(i);
        }

        // --- FEMORE DISTALE (triangoli di confronto) ---
        for (int i = 0; i < femurTriangles.Length; i += 3)
        {
            Vector3 aWorld = femurMeshFilter.transform.TransformPoint(
                femurVertices[femurTriangles[i]]);
            Vector3 bWorld = femurMeshFilter.transform.TransformPoint(
                femurVertices[femurTriangles[i + 1]]);
            Vector3 cWorld = femurMeshFilter.transform.TransformPoint(
                femurVertices[femurTriangles[i + 2]]);

            Vector3 aInPivot = femurPivot.InverseTransformPoint(aWorld);
            Vector3 bInPivot = femurPivot.InverseTransformPoint(bWorld);
            Vector3 cInPivot = femurPivot.InverseTransformPoint(cWorld);

            if (aInPivot.y < femurDistalY &&
                bInPivot.y < femurDistalY &&
                cInPivot.y < femurDistalY)
            {
                distalFemurTriangles.Add(femurTriangles[i]);
                distalFemurTriangles.Add(femurTriangles[i + 1]);
                distalFemurTriangles.Add(femurTriangles[i + 2]);
            }
        }

        Debug.Log($"Proximal tibia vertices: {proximalTibiaIndices.Count}");
        Debug.Log($"Distal femur triangles: {distalFemurTriangles.Count / 3}");
    }

    void Update()
    {
        for (int i = 0; i < colors.Length; i++)
            colors[i] = Color.clear;

        for (int k = 0; k < proximalTibiaIndices.Count; k += samplingStep)
        {
            int idx = proximalTibiaIndices[k];

            Vector3 p = transform.TransformPoint(tibiaVertices[idx]);
            Vector3 n = transform.TransformDirection(tibiaNormals[idx]).normalized;

            float minDist = float.MaxValue;
            Vector3 closestPoint = Vector3.zero;

            for (int t = 0; t < distalFemurTriangles.Count; t += 3)
            {
                Vector3 a = femurMeshFilter.transform.TransformPoint(
                    femurVertices[distalFemurTriangles[t]]);
                Vector3 b = femurMeshFilter.transform.TransformPoint(
                    femurVertices[distalFemurTriangles[t + 1]]);
                Vector3 c = femurMeshFilter.transform.TransformPoint(
                    femurVertices[distalFemurTriangles[t + 2]]);

                Vector3 cp;
                float d = PointTriangleDistanceAndClosest(p, a, b, c, out cp);

                if (d < minDist)
                {
                    minDist = d;
                    closestPoint = cp;
                }
            }

            if (minDist < contactThreshold)
            {
                Vector3 dir = (closestPoint - p).normalized;
                float dot = Vector3.Dot(n, dir);

                if (dot < normalDotThreshold)
                    colors[idx] = Color.red;
            }
        }

        tibiaMesh.colors = colors;
    }

    float PointTriangleDistanceAndClosest(
        Vector3 p, Vector3 a, Vector3 b, Vector3 c, out Vector3 closest)
    {
        Vector3 ab = b - a;
        Vector3 ac = c - a;
        Vector3 ap = p - a;

        float d1 = Vector3.Dot(ab, ap);
        float d2 = Vector3.Dot(ac, ap);
        if (d1 <= 0 && d2 <= 0)
        {
            closest = a;
            return Vector3.Distance(p, a);
        }

        Vector3 bp = p - b;
        float d3 = Vector3.Dot(ab, bp);
        float d4 = Vector3.Dot(ac, bp);
        if (d3 >= 0 && d4 <= d3)
        {
            closest = b;
            return Vector3.Distance(p, b);
        }

        float vc = d1 * d4 - d3 * d2;
        if (vc <= 0 && d1 >= 0 && d3 <= 0)
        {
            float v = d1 / (d1 - d3);
            closest = a + v * ab;
            return Vector3.Distance(p, closest);
        }

        Vector3 cp = p - c;
        float d5 = Vector3.Dot(ab, cp);
        float d6 = Vector3.Dot(ac, cp);
        if (d6 >= 0 && d5 <= d6)
        {
            closest = c;
            return Vector3.Distance(p, c);
        }

        float vb = d5 * d2 - d1 * d6;
        if (vb <= 0 && d2 >= 0 && d6 <= 0)
        {
            float w = d2 / (d2 - d6);
            closest = a + w * ac;
            return Vector3.Distance(p, closest);
        }

        float va = d3 * d6 - d5 * d4;
        if (va <= 0 && (d4 - d3) >= 0 && (d5 - d6) >= 0)
        {
            float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
            closest = b + w * (c - b);
            return Vector3.Distance(p, closest);
        }

        Vector3 n = Vector3.Cross(ab, ac).normalized;
        float dist = Mathf.Abs(Vector3.Dot(p - a, n));
        closest = p - Vector3.Dot(p - a, n) * n;
        return dist;
    }
}
