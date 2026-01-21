using System.Collections.Generic;
using UnityEngine;

[RequireComponent(typeof(MeshFilter))]
public class PhysicalSurfaceContact : MonoBehaviour
{
    [Header("Other bone")]
    public MeshFilter femurMeshFilter;

    [Header("Reference frames")]
    public Transform femurPivot;
    public Transform patellaPivot;

    [Header("Contact settings")]
    public float contactThreshold = 0.001f;   // 1 mm
    public float normalDotThreshold = -0.3f;  // normale patella verso femore
    public int samplingStep = 2;

    Mesh patellaMesh;
    Vector3[] patellaVertices;
    Vector3[] patellaNormals;
    Color[] colors;

    Vector3[] femurVertices;
    int[] femurTriangles;

    List<int> patellaContactVertices = new List<int>();
    List<int> femurContactTriangles = new List<int>();

    void Start()
    {
        patellaMesh = GetComponent<MeshFilter>().mesh;
        patellaVertices = patellaMesh.vertices;
        patellaNormals  = patellaMesh.normals;
        colors = new Color[patellaVertices.Length];

        femurVertices  = femurMeshFilter.mesh.vertices;
        femurTriangles = femurMeshFilter.mesh.triangles;

        PrecomputeRegions();
    }

    void PrecomputeRegions()
    {
        patellaContactVertices.Clear();
        femurContactTriangles.Clear();

        // --- PATELLA: vertici posteriori (normale verso femore) ---
        for (int i = 0; i < patellaVertices.Length; i++)
        {
            Vector3 nWorld = transform.TransformDirection(patellaNormals[i]).normalized;
            Vector3 toFemur = (femurPivot.position - transform.TransformPoint(patellaVertices[i])).normalized;

            if (Vector3.Dot(nWorld, toFemur) > 0.3f)
                patellaContactVertices.Add(i);
        }

        // --- FEMORE: triangoli anteriori (normale verso patella) ---
        for (int i = 0; i < femurTriangles.Length; i += 3)
        {
            Vector3 a = femurMeshFilter.transform.TransformPoint(femurVertices[femurTriangles[i]]);
            Vector3 b = femurMeshFilter.transform.TransformPoint(femurVertices[femurTriangles[i + 1]]);
            Vector3 c = femurMeshFilter.transform.TransformPoint(femurVertices[femurTriangles[i + 2]]);

            Vector3 triNormal = Vector3.Cross(b - a, c - a).normalized;
            Vector3 toPatella = (patellaPivot.position - (a + b + c) / 3f).normalized;

            if (Vector3.Dot(triNormal, toPatella) > 0.3f)
            {
                femurContactTriangles.Add(femurTriangles[i]);
                femurContactTriangles.Add(femurTriangles[i + 1]);
                femurContactTriangles.Add(femurTriangles[i + 2]);
            }
        }

        Debug.Log($"Patella contact vertices: {patellaContactVertices.Count}");
        Debug.Log($"Femur contact triangles: {femurContactTriangles.Count / 3}");
    }

    void Update()
    {
        for (int i = 0; i < colors.Length; i++)
            colors[i] = Color.clear;

        for (int k = 0; k < patellaContactVertices.Count; k += samplingStep)
        {
            int idx = patellaContactVertices[k];

            Vector3 p = transform.TransformPoint(patellaVertices[idx]);
            Vector3 n = transform.TransformDirection(patellaNormals[idx]).normalized;

            float minDist = float.MaxValue;
            Vector3 closest = Vector3.zero;

            for (int t = 0; t < femurContactTriangles.Count; t += 3)
            {
                Vector3 a = femurMeshFilter.transform.TransformPoint(femurVertices[femurContactTriangles[t]]);
                Vector3 b = femurMeshFilter.transform.TransformPoint(femurVertices[femurContactTriangles[t + 1]]);
                Vector3 c = femurMeshFilter.transform.TransformPoint(femurVertices[femurContactTriangles[t + 2]]);

                Vector3 cp;
                float d = PointTriangleDistanceAndClosest(p, a, b, c, out cp);

                if (d < minDist)
                {
                    minDist = d;
                    closest = cp;
                }
            }

            if (minDist < contactThreshold)
            {
                Vector3 dir = (closest - p).normalized;
                float dot = Vector3.Dot(n, dir);

                if (dot < normalDotThreshold)
                    colors[idx] = Color.red;
            }
        }

        patellaMesh.colors = colors;
    }

    // ---- distanza punto–triangolo (come nel tuo codice) ----
    float PointTriangleDistanceAndClosest(Vector3 p, Vector3 a, Vector3 b, Vector3 c, out Vector3 closest)
    {
        Vector3 ab = b - a;
        Vector3 ac = c - a;
        Vector3 ap = p - a;

        float d1 = Vector3.Dot(ab, ap);
        float d2 = Vector3.Dot(ac, ap);
        if (d1 <= 0 && d2 <= 0) { closest = a; return Vector3.Distance(p, a); }

        Vector3 bp = p - b;
        float d3 = Vector3.Dot(ab, bp);
        float d4 = Vector3.Dot(ac, bp);
        if (d3 >= 0 && d4 <= d3) { closest = b; return Vector3.Distance(p, b); }

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
        if (d6 >= 0 && d5 <= d6) { closest = c; return Vector3.Distance(p, c); }

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
