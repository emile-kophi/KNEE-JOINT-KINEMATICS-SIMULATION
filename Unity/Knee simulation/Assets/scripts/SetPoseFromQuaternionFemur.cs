using UnityEngine;

[ExecuteAlways]
public class SetPoseFromQuaternionFemur : MonoBehaviour
{
    [ContextMenu("Apply Frame 1 Pose")]
    void ApplyPose()
    {
        // Position from CSV (tx, ty, tz) in meters
        Vector3 position = new Vector3(
            0.164112f,
            0.435952f,
            0.395682f
        );

        // Quaternion from CSV (qx, qy, qz, qw)
        Quaternion rotation = new Quaternion(
            0.111383f,   // qx
           -0.52112f,    // qy
           -0.84092f,    // qz
            0.094214f    // qw
        );

        transform.SetPositionAndRotation(position, rotation);
    }
}
