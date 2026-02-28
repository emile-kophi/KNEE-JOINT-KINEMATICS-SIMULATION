using UnityEngine;

[RequireComponent(typeof(LineRenderer))]
public class MechanicalAxis : MonoBehaviour
{
    public Transform Prox;
    public Transform Dist;

    [Header("Axis length extension")]
    public float extension = 0.2f;   //anatomical axes extension

    private LineRenderer line;

    void Awake()
    {
        line = GetComponent<LineRenderer>();

        line.useWorldSpace = true;
        line.positionCount = 2;
        line.startWidth = 0.002f;
        line.endWidth = 0.002f;
    }

    void Update()
    {
        if (Prox == null || Dist == null) return;

        Vector3 direction = (Dist.position - Prox.position).normalized;

        Vector3 p0 = Prox.position - direction * extension;
        Vector3 p1 = Dist.position + direction * extension;

        line.SetPosition(0, p0);
        line.SetPosition(1, p1);
    }
}
