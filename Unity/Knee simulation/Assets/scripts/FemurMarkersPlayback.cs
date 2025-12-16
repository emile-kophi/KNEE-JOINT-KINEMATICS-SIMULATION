using UnityEngine;
using System.Collections.Generic;
using System.IO;
using System.Globalization;

public class FemurMarkersPlayback : MonoBehaviour
{
    public Transform femurAF;   // AF=TF (femoral head)
    public Transform femurLF;
    public Transform femurMF;

    [Header("CSV")]
    public string csvFileName = "femurMarkersForUnity.csv"; 
    public TextAsset csvTextOverride;
    public int motiveFrames = 1757;
    public float playbackFps = 100f;
    public bool loop = true;

    private Vector3[] AF, LF, MF;
    private float t;
    private int N;

    void Start()
    {
        string rawText = null;

        if (csvTextOverride != null)
        {
            rawText = csvTextOverride.text;
        }
        else
        {
            string filePath = Path.Combine(Application.streamingAssetsPath, csvFileName);
            if (!File.Exists(filePath))
            {
                Debug.LogError("Femur CSV markers not found: " + filePath);
                enabled = false;
                return;
            }
            rawText = File.ReadAllText(filePath);
        }

        ParseCsv(rawText);
        Debug.Log("FemurMarkersPlayback: loaded " + N + " frames.");
    }

    void ParseCsv(string text)
    {
        var ci = CultureInfo.InvariantCulture;
        var lines = text.Split('\n');

        var af = new List<Vector3>();
        var lf = new List<Vector3>();
        var mf = new List<Vector3>();

        // skip header
        for (int i = 1; i < lines.Length; i++)
        {
            var line = lines[i].Trim();
            if (string.IsNullOrEmpty(line)) continue;

            var v = line.Split(',');
            if (v.Length < 10) continue;

            float P(string s) => float.Parse(s, ci);

            Vector3 AF_world = new Vector3(P(v[1]), P(v[2]), P(v[3]));
            Vector3 LF_world = new Vector3(P(v[4]), P(v[5]), P(v[6]));
            Vector3 MF_world = new Vector3(P(v[7]), P(v[8]), P(v[9]));

            af.Add(AF_world);
            lf.Add(LF_world);
            mf.Add(MF_world);
        }

        AF = af.ToArray();
        LF = lf.ToArray();
        MF = mf.ToArray();

        N = AF.Length;
    }

    void Update()
    {
        if (N == 0) return;

        t += Time.deltaTime;
        int frame = Mathf.FloorToInt(t * playbackFps);
        int max = Mathf.Min(motiveFrames - 1, N - 1);
        frame = loop ? frame % (max + 1) : Mathf.Clamp(frame, 0, max);
        int k = Mathf.Clamp(frame, 0, N - 1);

        if (femurAF) femurAF.position = AF[k];
        if (femurLF) femurLF.position = LF[k];
        if (femurMF) femurMF.position = MF[k];
    }
}
