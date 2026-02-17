using UnityEngine;

public class F_T_ContactVisualizer : MonoBehaviour
{
    [Header("Bone Assignments")]
    public Transform tibiaObject;
    public Transform pointMediale;
    public Transform pointLaterale;

    [Header("Biomechanics Settings")]
    public float condyleRadius = 0.025f;

    [Header("Visual Settings")]
    public float dotSize = 0.01f;
    public float trailDuration = 0.5f;
    public Color colorMedial = Color.cyan;
    public Color colorLateral = Color.magenta;

    [Header("Contact Threshold Settings")]
    public float maxScanDistance = 0.1f;
    public float contactThreshold = 0.005f;

    private ContactHelper helperMediale;
    private ContactHelper helperLaterale;

    void Start()
    {
        helperMediale = new ContactHelper(pointMediale, colorMedial, dotSize, trailDuration);
        helperLaterale = new ContactHelper(pointLaterale, colorLateral, dotSize, trailDuration);
    }

    void Update()
    {
        helperMediale.CheckContact(tibiaObject, maxScanDistance, contactThreshold, condyleRadius);
        helperLaterale.CheckContact(tibiaObject, maxScanDistance, contactThreshold, condyleRadius);
    }

    void OnDrawGizmos()
    {
        if (pointMediale != null)
        {
            Gizmos.color = colorMedial;
            Gizmos.DrawWireSphere(pointMediale.position, condyleRadius);
        }

        if (pointLaterale != null)
        {
            Gizmos.color = colorLateral;
            Gizmos.DrawWireSphere(pointLaterale.position, condyleRadius);
        }
    }

    // =========================
    // ===== HELPER CLASS =====
    // =========================

    private class ContactHelper
    {
        private Transform originPoint;
        private GameObject sphereObj;
        private TrailRenderer trail;
        private bool wasHitLastFrame = false;

        public ContactHelper(Transform origin, Color c, float size, float time)
        {
            originPoint = origin;

            // Create visualizer sphere
            sphereObj = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            sphereObj.name = origin.name + "_Viz";
            Object.Destroy(sphereObj.GetComponent<Collider>());

            // --- LAYER ASSIGNMENT ---
            // Assigns "ContactHidden" layer so it can be culled by specific cameras (e.g., Display 5)
            int layerID = LayerMask.NameToLayer("ContactHidden");
            if (layerID != -1)
            {
                sphereObj.layer = layerID;
            }
            else
            {
                Debug.LogWarning("Layer 'Contact' not found. Please add it in Edit -> Project Settings -> Tags and Layers.");
            }

            // Visual setup
            Material mat = new Material(Shader.Find("Sprites/Default"));
            mat.color = c;

            sphereObj.GetComponent<Renderer>().material = mat;
            sphereObj.transform.localScale = Vector3.one * size;

            // Trail setup
            trail = sphereObj.AddComponent<TrailRenderer>();
            trail.startWidth = size;
            trail.endWidth = 0f;
            trail.time = time;
            trail.material = mat;
            trail.minVertexDistance = 0.005f;

            sphereObj.SetActive(false);
        }

        public void CheckContact(Transform targetTibia, float maxDist, float threshold, float radius)
        {
            RaycastHit probeHit;

            // STEP 1 — Find tibial surface and normal
            if (Physics.Raycast(originPoint.position, -targetTibia.up, out probeHit, maxDist))
            {
                if (probeHit.collider.transform == targetTibia || probeHit.collider.transform.IsChildOf(targetTibia))
                {
                    Vector3 contactDirection = -probeHit.normal;
                    RaycastHit sphereHit;

                    // STEP 2 — SphereCast along surface normal
                    if (Physics.SphereCast(originPoint.position, radius, contactDirection, out sphereHit, maxDist))
                    {
                        if (sphereHit.collider.transform == targetTibia || sphereHit.collider.transform.IsChildOf(targetTibia))
                        {
                            if (sphereHit.distance <= threshold)
                            {
                                // Contact detected
                                sphereObj.SetActive(true);
                                sphereObj.transform.position = sphereHit.point;

                                trail.emitting = true;
                                wasHitLastFrame = true;
                            }
                            else
                            {
                                DisableContact();
                            }
                            return;
                        }
                    }
                }
            }
            DisableContact();
        }

        private void DisableContact()
        {
            if (wasHitLastFrame)
            {
                trail.emitting = false;
                sphereObj.SetActive(false);
            }
            wasHitLastFrame = false;
        }
    }
}