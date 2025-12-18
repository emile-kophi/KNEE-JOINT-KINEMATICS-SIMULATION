using UnityEngine;

public class DebugMeshOffset : MonoBehaviour
{
    void Start()
    {
        var mf = GetComponent<MeshFilter>();
        if (mf == null)
        {
            Debug.Log("No MeshFilter on this GameObject.");
            return;
        }

        Bounds b = mf.sharedMesh.bounds; // local mesh bounds
        Debug.Log($"Mesh bounds center (local): {b.center}");
        Debug.Log($"Mesh bounds size (local): {b.size}");
    }
}
