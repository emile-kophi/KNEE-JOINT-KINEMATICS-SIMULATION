using System.Collections.Generic;
using UnityEngine;
using System.IO;
using System.Globalization;

public class CSVAnimator : MonoBehaviour
{
    [Header("CSV Settings")]
    public string csvFileName;     // e.g. tibiaTransformForUnity.csv
    public float frameRate = 100f;
    public bool loop = true;

    private readonly List<Vector3> positions = new();
    private readonly List<Quaternion> rotations = new();

    private float timer;
    private int currentFrame;

    void Start()
    {
        string filePath = Path.Combine(Application.streamingAssetsPath, csvFileName);
        if (!File.Exists(filePath))
        {
            Debug.LogError($"CSV not found: {filePath}");
            enabled = false;
            return;
        }

        var ci = CultureInfo.InvariantCulture;

        using (var reader = new StreamReader(filePath))
        {
            reader.ReadLine(); // Skip header

            while (!reader.EndOfStream)
            {
                var line = reader.ReadLine();
                if (string.IsNullOrWhiteSpace(line))
                    continue;

                var v = line.Split(',');
                if (v.Length < 8)
                    continue;

                // CSV FORMAT (UNITY-READY):
                // frame, qx, qy, qz, qw, tx, ty, tz

                float qx = float.Parse(v[1], ci);
                float qy = float.Parse(v[2], ci);
                float qz = float.Parse(v[3], ci);
                float qw = float.Parse(v[4], ci);

                Quaternion q = new Quaternion(qx, qy, qz, qw);

                float x = float.Parse(v[5], ci);
                float y = float.Parse(v[6], ci);
                float z = float.Parse(v[7], ci);

                Vector3 p = new Vector3(x, y, z);

                rotations.Add(q);
                positions.Add(p);
            }
        }

        // Initialize at frame 0
        if (positions.Count > 0)
        {
            transform.SetPositionAndRotation(positions[0], rotations[0]);
        }
    }

    void Update()
    {
        if (positions.Count == 0)
            return;

        timer += Time.deltaTime;
        int targetFrame = Mathf.FloorToInt(timer * frameRate);

        if (loop)
            targetFrame %= positions.Count;
        else
            targetFrame = Mathf.Min(targetFrame, positions.Count - 1);

        currentFrame = targetFrame;

        transform.SetPositionAndRotation(
            positions[currentFrame],
            rotations[currentFrame]
        );
    }
}
