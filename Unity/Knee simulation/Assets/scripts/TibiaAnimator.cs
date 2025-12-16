using System.Collections.Generic;
using UnityEngine;
using System.IO;
using System.Globalization;

public class TibiaAnimator : MonoBehaviour
{
    [Header("CSV")]
    public string csvFileName = "tibiaTransformForUnity.csv";
    public float frameRate = 100f;
    public bool loop = true;

    [Header("Offset (applied after CSV)")]
    public Vector3 rotationOffsetEuler;   // opzionale
    public Vector3 positionOffsetLocal;   // opzionale

    private readonly List<Vector3> positions = new();
    private readonly List<Quaternion> rotations = new();

    private int currentFrame = 0;
    private float timer;

    private Quaternion rotationOffset;
    private Vector3 positionOffset;

    void Start()
    {
        rotationOffset = Quaternion.Euler(rotationOffsetEuler);
        positionOffset = positionOffsetLocal;

        string filePath = Path.Combine(Application.streamingAssetsPath, csvFileName);
        if (!File.Exists(filePath))
        {
            Debug.LogError("CSV not found: " + filePath);
            enabled = false;
            return;
        }

        var ci = CultureInfo.InvariantCulture;

        using (var reader = new StreamReader(filePath))
        {
            reader.ReadLine(); // header

            while (!reader.EndOfStream)
            {
                var line = reader.ReadLine();
                if (string.IsNullOrWhiteSpace(line))
                    continue;

                var v = line.Split(',');
                if (v.Length < 8)
                    continue;

                // === Quaternion (qx qy qz qw) ===
                float qx = float.Parse(v[1], ci);
                float qy = float.Parse(v[2], ci);
                float qz = float.Parse(v[3], ci);
                float qw = float.Parse(v[4], ci);

                // === Position (x y z) ===
                float x  = float.Parse(v[5], ci);
                float y  = float.Parse(v[6], ci);
                float z  = float.Parse(v[7], ci);

                rotations.Add(new Quaternion(qx, qy, qz, qw));
                positions.Add(new Vector3(x, y, z));
            }
        }

        // Set start pose
        if (positions.Count > 0)
        {
            transform.position = positions[0];
            transform.rotation = rotations[0];
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

        Vector3 posCSV = positions[currentFrame];
        Quaternion rotCSV = rotations[currentFrame];

        // apply optional offsets
        transform.position = posCSV + rotCSV * positionOffset;
        transform.rotation = rotCSV * rotationOffset;
    }
}
