using UnityEngine;

public class DebugMeshOffset : MonoBehaviour
{
    private MeshFilter mf;
    private Renderer rend;

    void Start()
    {
        mf = GetComponent<MeshFilter>();
        rend = GetComponent<Renderer>();

        if (mf == null)
            Debug.LogError("MeshFilter not found on this GameObject.");

        if (rend == null)
            Debug.LogError("Renderer not found on this GameObject.");
    }

    void Update()
    {
        if (mf == null || rend == null)
            return;

        // Transform info (what you animate from CSV)
        Vector3 tPos = transform.position;
        Quaternion tRot = transform.rotation;

        // Mesh bounds in LOCAL space (relative to mesh origin)
        Vector3 meshLocalCenter = mf.sharedMesh.bounds.center;

        // Renderer bounds in WORLD space (actual geometry position)
        Vector3 meshWorldCenter = rend.bounds.center;

        Debug.Log(
            $"[{name}] TRANSFORM world pos = {tPos}, rot = {tRot.eulerAngles}\n" +
            $"[{name}] MESH bounds LOCAL center = {meshLocalCenter}\n" +
            $"[{name}] MESH bounds WORLD center = {meshWorldCenter}\n" +
            $"[{name}] WORLD offset (meshCenter - transform) = {(meshWorldCenter - tPos)}"
        );
    }
}
