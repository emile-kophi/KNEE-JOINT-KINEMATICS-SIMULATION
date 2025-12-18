using UnityEngine;

public class ExportLocalLandmarks : MonoBehaviour
{
    public Transform p1;
    public Transform p2;
    public Transform p3;

    void Start()
    {
        Vector3 L1 = transform.InverseTransformPoint(p1.position);
        Vector3 L2 = transform.InverseTransformPoint(p2.position);
        Vector3 L3 = transform.InverseTransformPoint(p3.position);

        Debug.Log($"{name} local points:");
        Debug.Log($"P1: {L1.x}, {L1.y}, {L1.z}");
        Debug.Log($"P2: {L2.x}, {L2.y}, {L2.z}");
        Debug.Log($"P3: {L3.x}, {L3.y}, {L3.z}");
    }
}
