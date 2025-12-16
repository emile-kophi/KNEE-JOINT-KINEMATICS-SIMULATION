using UnityEngine;
using System.Collections.Generic;
using System.IO;
using System.Globalization;

public class TibiaMarkersPlayback : MonoBehaviour
{
    public Transform tibiaAMT;
    public Transform tibiaLT;
    public Transform tibiaMT;

    [Header("CSV")]
    public string csvFileName = "tibiaMarkersForUnity.csv"; 
    public TextAsset csvTextOverride;
    public int motiveFrames = 1757;
    public float playbackFps = 100f;
    public bool loop = true;

    private Vector3[] AMT, LT, MT;
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
                Debug.LogError("Tibia CSV markers not found: " + filePath);
                enabled = false;
                return;
            }
            rawText = File.ReadAllText(filePath);
        }

        ParseCsv(rawText);
        Debug.Log("TibiaMarkersPlayback: loaded " + N + " frames.");
    }

    void ParseCsv(string text)
    {
        var ci = CultureInfo.InvariantCulture;
        var lines = text.Split('\n');

        var amt = new List<Vector3>();
        var lt  = new List<Vector3>();
        var mt  = new List<Vector3>();

        // skip header
        for (int i = 1; i < lines.Length; i++)
        {
            var line = lines[i].Trim();
            if (string.IsNullOrEmpty(line)) continue;

            var v = line.Split(',');
            if (v.Length < 10) continue;

            float P(string s) => float.Parse(s, ci);

            Vector3 AMT_world = new Vector3(P(v[1]), P(v[2]), P(v[3]));
            Vector3 LT_world  = new Vector3(P(v[4]), P(v[5]), P(v[6]));
            Vector3 MT_world  = new Vector3(P(v[7]), P(v[8]), P(v[9]));

            amt.Add(AMT_world);
            lt.Add(LT_world);
            mt.Add(MT_world);
        }

        AMT = amt.ToArray();
        LT  = lt.ToArray();
        MT  = mt.ToArray();

        N = AMT.Length;
    }

    void Update()
    {
        if (N == 0) return;

        t += Time.deltaTime;
        int frame = Mathf.FloorToInt(t * playbackFps);
        int max = Mathf.Min(motiveFrames - 1, N - 1);
        frame = loop ? frame % (max + 1) : Mathf.Clamp(frame, 0, max);
        int k = Mathf.Clamp(frame, 0, N - 1);
        
        if (tibiaAMT) tibiaAMT.position = AMT[k];
        if (tibiaLT)  tibiaLT.position  = LT[k];
        if (tibiaMT)  tibiaMT.position  = MT[k];
    }
}
