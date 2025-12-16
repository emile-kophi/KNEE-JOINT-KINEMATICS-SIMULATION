using UnityEngine;

public class ExportBlenderMarkers : MonoBehaviour
{
    [Header("Landmark mesh (Blender)")]
    public Transform AMT_blender;
    public Transform LT_blender;
    public Transform MT_blender;

    [Header("Marker reali (Unity/Motive)")]
    public Transform TibiaAMT;
    public Transform TibiaLT;
    public Transform TibiaMT;

    void Update()
    {
        int frame = Time.frameCount;

        // Punti sulla mesh (figli della mesh)
        Vector3 pAMT = AMT_blender.position;
        Vector3 pLT  = LT_blender.position;
        Vector3 pMT  = MT_blender.position;

        // Marker reali/mocap (gameobject animati)
        Vector3 mAMT = TibiaAMT != null ? TibiaAMT.position : Vector3.zero;
        Vector3 mLT  = TibiaLT  != null ? TibiaLT.position  : Vector3.zero;
        Vector3 mMT  = TibiaMT  != null ? TibiaMT.position  : Vector3.zero;

        Debug.Log(
            $"Frame {frame} | " +
            $"Mesh AMT: {pAMT:F6} | LT: {pLT:F6} | MT: {pMT:F6} || " +
            $"Motive AMT: {mAMT:F6} | LT: {mLT:F6} | MT: {mMT:F6}"
        );
    }
}
