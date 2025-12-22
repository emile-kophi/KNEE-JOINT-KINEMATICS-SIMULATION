using UnityEngine;

[ExecuteAlways]
public class SetPoseFromQuaternionTibia : MonoBehaviour
{
    [ContextMenu("Apply Frame 0 Pose")]
    void ApplyPose()
    {
        // Position from CSV
        Vector3 position = new Vector3(
            0.142055134f,
            0.327164581f,
            0.120450753f
        );

        // Quaternion from CSV (qx, qy, qz, qw)
        Quaternion rotation = new Quaternion(
            0.231364736f,   // qx
           -0.144607585f,   // qy
           -0.048197868f,   // qz
            0.96085169f     // qw
        );

        transform.SetPositionAndRotation(position, rotation);
    }
}
