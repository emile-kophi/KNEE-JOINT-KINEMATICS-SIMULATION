using UnityEngine;
using System.Collections.Generic;

public class PatellaFemurContact : MonoBehaviour
{
    [Header("1. Collegamenti")]
    public MeshFilter patellaMesh;
    public Collider femurCollider; // Il Mesh Collider del femore

    [Header("2. Impostazioni Semplici")]
    public float contactDistance = 0.003f; // Distanza massima (spessore cartilagine)
    public float sphereSize = 0.003f;      // Grandezza pallini
    
    [Tooltip("Direzione del raggio. (0,0,-1) è solitamente 'Indietro' verso il femore.")]
    public Vector3 castDirection = new Vector3(0, 0, -1); 

    [Header("3. Colore Contatto")]
    public Color contactColor = Color.red; // Colore delle sfere quando toccano

    [Header("4. Performance")]
    [Range(1, 50)]
    public int precisionStep = 10; // 1 = Tutti i vertici (Lento), 10 = Veloce

    [Header("5. Debug Visuale (Raggi)")]
    public bool showRays = true;          // Spunta per vedere le linee
    public Color rayColor = Color.yellow; // Scegli il colore delle linee dei raggi

    // Memoria interna (non toccare)
    private List<GameObject> spherePool = new List<GameObject>();
    private List<Vector3> activeVertices = new List<Vector3>();
    private Material sharedMaterial;
    private GameObject container;
    private int targetLayer; // Variabile per memorizzare l'ID del layer

    void Start()
    {
        if (!patellaMesh || !femurCollider) return;

        // 1. Trova l'ID del Layer "ContactF_P_Hidden"
        targetLayer = LayerMask.NameToLayer("ContactF_P_Hidden");
        
        // Controllo di sicurezza: se il layer non esiste, avvisa ma continua (usando Default)
        if (targetLayer == -1)
        {
            Debug.LogWarning("⚠️ ATTENZIONE: Il layer 'ContactF_P_Hidden' non esiste nelle impostazioni del progetto! Le sfere useranno il layer Default.");
            targetLayer = 0; // Layer Default
        }

        // 2. Creiamo un materiale sicuro che non diventa VIOLA
        sharedMaterial = new Material(Shader.Find("Sprites/Default"));
        sharedMaterial.color = contactColor;

        // 3. Generiamo il pool di sfere (spente)
        GeneratePool();
    }

    void GeneratePool()
    {
        if (container) Destroy(container);
        container = new GameObject("Contact_Spheres_Pool");
        container.transform.SetParent(this.transform);

        Vector3[] verts = patellaMesh.sharedMesh.vertices;
        spherePool.Clear();
        activeVertices.Clear();

        // Salviamo solo i vertici che useremo (in base allo step)
        for (int i = 0; i < verts.Length; i += precisionStep)
        {
            activeVertices.Add(verts[i]);

            GameObject s = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            Destroy(s.GetComponent<Collider>()); // Via la fisica
            
            // --- ASSEGNAZIONE DEL LAYER ---
            s.layer = targetLayer; 
            // -----------------------------

            s.transform.SetParent(container.transform);
            s.transform.localScale = Vector3.one * sphereSize;
            s.GetComponent<Renderer>().sharedMaterial = sharedMaterial; // Assegna materiale
            s.SetActive(false); // Partono spente
            spherePool.Add(s);
        }
    }

    void Update()
    {
        // Aggiorna il colore live se lo cambi in inspector
        if (sharedMaterial.color != contactColor) sharedMaterial.color = contactColor;
        
        CheckContacts();
    }

    void CheckContacts()
    {
        Transform t = patellaMesh.transform;
        
        // Calcoliamo la direzione del raggio nel mondo reale
        Vector3 worldDir = t.TransformDirection(castDirection).normalized;
        float checkDistance = 0.05f; // Lunghezza del controllo (5cm)

        for (int i = 0; i < activeVertices.Count; i++)
        {
            Vector3 startPoint = t.TransformPoint(activeVertices[i]);
            
            // Arretriamo leggermente il punto di partenza
            Vector3 origin = startPoint - (worldDir * 0.01f); 

            // --- DISEGNA IL RAGGIO PER DEBUG ---
            if (showRays)
            {
                // Disegna una linea dall'origine verso la direzione
                Debug.DrawRay(origin, worldDir * checkDistance, rayColor);
            }
            // -----------------------------------

            Ray ray = new Ray(origin, worldDir);
            RaycastHit hit;

            // Spara il raggio
            if (femurCollider.Raycast(ray, out hit, checkDistance)) 
            {
                // Calcola distanza reale tra superficie rotula e punto colpito
                float distance = Vector3.Distance(startPoint, hit.point);

                if (distance <= contactDistance)
                {
                    spherePool[i].SetActive(true);
                    
                    // --- OPZIONE A: Sfere sulla PATELLA ---
                    // La sfera rimane sul vertice della Rotula
                    spherePool[i].transform.position = startPoint; 
                    // --------------------------------------
                }
                else
                {
                    spherePool[i].SetActive(false);
                }
            }
            else
            {
                spherePool[i].SetActive(false);
            }
        }
    }
}