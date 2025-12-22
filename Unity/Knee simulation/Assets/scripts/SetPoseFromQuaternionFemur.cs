using UnityEngine;

[ExecuteAlways]
public class SetPoseFromQuaternionFemur : MonoBehaviour
{
    [ContextMenu("Apply Frame 0 Pose")]
    void ApplyPose()
    {
        // Position from CSV
        Vector3 position = new Vector3(
            0.163516f,
            0.435236f,
            0.380094f
        );

        // Quaternion from CSV (qx, qy, qz, qw)
        Quaternion rotation = new Quaternion(
           -0.97173f,   // qx
            0.197954f,  // qy
            0.052669f,  // qz
            0.11736f    // qw
        );

        transform.SetPositionAndRotation(position, rotation);
    }
}
