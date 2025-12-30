using UnityEngine;

[ExecuteAlways]
public class SetPoseFromQuaternionPatella : MonoBehaviour
{
    [ContextMenu("Apply Frame 1 Pose")]
    void ApplyPose()
    {
        // Position from CSV (tx, ty, tz)
        Vector3 position = new Vector3(
            0.150661f,
            0.514328f,
            0.168941f
        );

        // Quaternion from CSV (qx, qy, qz, qw)
        Quaternion rotation = new Quaternion(
           -0.30559f,   // qx
            0.294484f,  // qy
           -0.90486f,   // qz
            0.033611f   // qw
        );

        transform.SetPositionAndRotation(position, rotation);
    }
}
