using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Burst;
using Unity.Jobs;
using UnityEngine.Jobs;
using Unity.Collections;
using TMPro;


[BurstCompile(FloatPrecision.Standard, FloatMode.Fast)]
struct EntityMovementJob : IJobParallelForTransform
{
    [ReadOnly] public NativeArray<bool> entitySides;
    [ReadOnly] public NativeArray<float> influenceMap;
    [ReadOnly] public NativeArray<int> offsets;
    [ReadOnly] public int ROWS;
    [ReadOnly] public int COLS;
    [ReadOnly] public float moveSpeed;
    [ReadOnly] public float deltaTime;


    public void Execute(int index, TransformAccess transform)
    {
        float x = transform.position.x;
        float y = transform.position.z;

        // Truncated x and y coordinates
        int intX = Mathf.RoundToInt(x);
        int intY = Mathf.RoundToInt(y);


        bool side = entitySides[index];

        int moveIndex = intX * ROWS + intY; // Default move is the current position
        float highestInfluence = influenceMap[moveIndex];

        for (int i = 0; i < offsets.Length; i += 2)
        {
            int xPos = intX + offsets[i];
            int yPos = intY + offsets[i + 1];

            if(xPos < 0 || xPos >= COLS
                || yPos < 0 || yPos >= ROWS)
            {
                continue;
            }
            else
            {
                int mapIndex = xPos * ROWS + yPos;

                if (mapIndex >= 0 && mapIndex < influenceMap.Length)
                {
                    float influence = influenceMap[mapIndex];
                    if (side)
                    {
                        if (influence > highestInfluence)
                        {
                            highestInfluence = influence;
                            moveIndex = mapIndex;
                        }
                    }
                    else
                    {
                        if (influence < highestInfluence)
                        {
                            highestInfluence = influence;
                            moveIndex = mapIndex;
                        }
                    }
                }
            }
        }

        float moveX = moveIndex / ROWS;
        float moveY = moveIndex % ROWS;

        Vector3 moveVector = new Vector3(moveX - intX, 0.0f, moveY - intY);
        float modifiedMoveSpeed = 0;
        if (moveSpeed > 0)
        {
            modifiedMoveSpeed = moveSpeed + Mathf.Abs(highestInfluence) * 5;
        }

        //transform.position += moveVector * moveSpeed * deltaTime;
        transform.position += moveVector * modifiedMoveSpeed * deltaTime;
    }
}

public class EntityManager : MonoBehaviour
{
    public static EntityManager Instance;

    [SerializeField] GameObject entityPrefab;

    List<GameObject> entityList;

    EntityMovementJob entityMovementJob;
    JobHandle movementJobHandle;


    TransformAccessArray transformAccessArray;
    List<Transform> entityTransformList;

    List<bool> entitySidesList;
    NativeArray<bool> entitySidesArray;
    NativeArray<int> offsetArray;
    public NativeArray<float> influenceMap;


    [SerializeField] float entityMoveSpeed = 2.0f;


    [SerializeField] float radius = 1.0f;
    float mouseScroll;

    [SerializeField] TMP_Text mousePointRadiusText;
    [SerializeField] TMP_Text entityCountText;
    [SerializeField] TMP_Text movementUpdateTimeText;
    [SerializeField] TMP_Text moveSpeedText;
    [ReadOnly] int[] offsets = new int[] { 0, 1, 1, 1, 1, 0, 1, -1, 0, -1, -1, -1, -1, 0, -1, 1};

    int ROWS = 320;
    int COLS = 320;

    bool entitiesChanged = true;
    Timer timer;

    

    private void Awake()
    {
        if (Instance != null && Instance != this)
        {
            Destroy(this);
        }
        else
        {
            Instance = this;
        }
    }
    // Start is called before the first frame update
    void Start()
    {
        ROWS = InfluenceMap.Instance.GetMapRows();
        COLS = InfluenceMap.Instance.GetMapCols();

        entityList = new List<GameObject>();
        entitySidesList = new List<bool>();

        offsets = new int[] { 0, 1, 1, 1, 1, 0, 1, -1, 0, -1, -1, -1, -1, 0, -1, 1 };
        offsetArray = new NativeArray<int>(offsets, Allocator.Persistent);
        entityTransformList = new List<Transform>();
        transformAccessArray = new TransformAccessArray();
        timer = new Timer();

        mousePointRadiusText.text = "Mouse point radius: " + radius.ToString();
        moveSpeedText.text = "Move speed: " + entityMoveSpeed;
    }

    // Update is called once per frame
    void Update()
    {

        if (influenceMap.IsCreated && influenceMap.Length > 0)
        {
            if (entityList.Count > 0)
            {
                //EntityMovementMT();
                
                EntityMovement();
            }
        }


        mouseScroll = Input.mouseScrollDelta.y;
        if (mouseScroll != 0.0f)
        {
            radius = (radius + mouseScroll > 0.0f) ? radius += mouseScroll : radius = 0;
            mousePointRadiusText.text = "Mouse point radius: " + radius.ToString();
        }

        if(Input.GetKeyDown(KeyCode.UpArrow))
        {
            entityMoveSpeed += 1.0f;
            moveSpeedText.text = "Move speed: " + entityMoveSpeed;
        }

        if (Input.GetKeyDown(KeyCode.DownArrow))
        {
            entityMoveSpeed = entityMoveSpeed - 1.0f >= 0 ? entityMoveSpeed - 1.0f : 0.0f;
            moveSpeedText.text = "Move speed: " + entityMoveSpeed;
        }


        if (Input.GetMouseButtonDown(0))
        {
            SpawnEntitiesCircle(Mathf.RoundToInt(radius));
        }

        if (Input.GetMouseButton(0) && Input.GetKey(KeyCode.LeftControl))
        {
            SpawnEntitiesCircle(Mathf.RoundToInt(radius));
        }

        if (Input.GetKeyDown(KeyCode.Backspace))
        {
            ClearEntities();
        }

        if (Input.GetMouseButtonDown(1))
        {
            SpawnRandomEntities(5000);
        }

        entityCountText.text = "Entities: " + entityList.Count;

    }

    void EntityMovement()
    {
        timer.Restart();

        // K‰yd‰‰n l‰pi kaikkien olioiden Transform-komponentit.
        for (int i = 0; i < entityTransformList.Count; i++)
        {
            Transform t = entityTransformList[i];

            float x = t.position.x;
            float y = t.position.z;

            // Pyˆristet‰‰n koordinaatit kokonaisluvuiksi vaikutuskartan indeksointia varten.
            int intX = Mathf.RoundToInt(x);
            int intY = Mathf.RoundToInt(y);

            // Haetaan olion puoli.
            bool side = entitySidesList[i];

            int moveIndex = intX * ROWS + intY; // Vakiona liikutaan nykyiseen ruutuun.

            // Asetetaan korkein vaikutus nykyisen ruudun mukaan.
            float highestInfluence = influenceMap[moveIndex];

            // K‰yd‰‰n l‰pi kaikki ruudut v‰littˆm‰ss‰ l‰heisyydess‰
            // ja valitaan ruutu jolla on korkein vaikutus olion puolta kohtaan.
            for (int j = 0; j < offsets.Length; j += 2)
            {
                int xPos = intX + offsets[j];
                int yPos = intY + offsets[j + 1];

                // Tarkistetaan ett‰ koordinaatit ovat kartan sis‰ll‰.
                if (xPos < 0 || xPos >= COLS
                    || yPos < 0 || yPos >= ROWS)
                {
                    continue;
                }
                else
                {
                    // Lasketaan koordinaatteja vastaava taulukon indeksi
                    // ja haetaan sit‰ vastaava vaikutusarvo vaikutuskartasta.
                    int mapIndex = xPos * ROWS + yPos;

                    float influence = influenceMap[mapIndex];

                    // P‰ivitet‰‰n korkein vaikutusarvo ja liikkumis indeksi puolen mukaan.
                    if (side)
                    {
                        if (influence > highestInfluence)
                        {
                            highestInfluence = influence;
                            moveIndex = mapIndex;
                        }
                    }
                    else
                    {
                        if (influence < highestInfluence)
                        {
                            highestInfluence = influence;
                            moveIndex = mapIndex;
                        }
                    }

                }
            }

            // Muunnetaan yksiulotteinen taulukon indeksi x- ja y-koordinaateiksi.
            float moveX = moveIndex / ROWS;
            float moveY = moveIndex % ROWS;

            // Lasketaan liikkumis suunta nykyisest‰ ruudusta.
            Vector3 moveVector = new Vector3(moveX - intX, 0.0f, moveY - intY);

            float modifiedMoveSpeed = 0;
            if (entityMoveSpeed > 0)
            {
                // Muunnetaan liikkumisnopeutta korkeimman vaikutuksen mukaan.
                modifiedMoveSpeed = entityMoveSpeed + Mathf.Abs(highestInfluence) * 5;
            }

            // P‰ivitet‰‰n Transform-komponentin sijaintia liikkumissuunnan ja nopeuden mukaan.
            t.position += moveVector * modifiedMoveSpeed * Time.deltaTime;
        }
        timer.Stop();
        movementUpdateTimeText.text = "Movement update time: " + timer.GetTimeStr();
    }

    public void EntityMovementMT()
    {
        timer.Restart();
        entityMovementJob = new EntityMovementJob();

        if(entitiesChanged)
        {
            if (entitySidesArray.IsCreated)
            {
                entitySidesArray.Dispose();
            }

            entitySidesArray = new NativeArray<bool>(entitySidesList.ToArray(), Allocator.Persistent);

            if (transformAccessArray.isCreated)
            {
                transformAccessArray.Dispose();
            }

            transformAccessArray = new TransformAccessArray(entityTransformList.Count);
            transformAccessArray.SetTransforms(entityTransformList.ToArray());

            entitiesChanged = false;
        }
        
        
        entityMovementJob.entitySides = entitySidesArray;
        entityMovementJob.influenceMap = this.influenceMap;
        entityMovementJob.COLS = this.COLS;
        entityMovementJob.ROWS = this.ROWS;
        entityMovementJob.offsets = offsetArray;
        entityMovementJob.moveSpeed = entityMoveSpeed;
        entityMovementJob.deltaTime = Time.deltaTime;
        movementJobHandle = entityMovementJob.Schedule(transformAccessArray);
        JobHandle.ScheduleBatchedJobs();

        timer.Stop();
        movementUpdateTimeText.text = "Movement update time: " + timer.GetTimeStr();

    }

    private void LateUpdate()
    {
        movementJobHandle.Complete();
        UpdateInfluenceMap();
    }

    GameObject SpawnEntity()
    {
        GameObject newEntity = Instantiate(entityPrefab);
        entityList.Add(newEntity);
        bool side = Random.value > 0.5f;

        if(side)
        {
            newEntity.GetComponent<Renderer>().material.color = Color.cyan;
        }
        else
        {
            newEntity.GetComponent<Renderer>().material.color = Color.yellow;
        }

        entitySidesList.Add(side);
        entityTransformList.Add(newEntity.transform);
        //transformAccessArray.Add(newEntity.transform);
        entitiesChanged = true;
        return newEntity;
    }

    void SpawnRandomEntities(int count)
    {
        for (int i = 0; i < count; i++)
        {
            GameObject newEntity = SpawnEntity();
            Vector3 entityPos = new Vector3(Random.Range(0, COLS), 0, Random.Range(0, ROWS));
            newEntity.transform.position = entityPos;
            newEntity.transform.localScale = new Vector3(0.1f, 0.1f, 0.1f);
        }
    }

    void SpawnEntitiesCircle(int radius)
    {
        Ray ray = Camera.main.ScreenPointToRay(Input.mousePosition);
        RaycastHit hit;

        if (Physics.Raycast(ray, out hit))
        {
            for (int y = -radius; y <= radius; y++)
            {
                for (int x = -radius; x <= radius; x++)
                {
                    if (x * x + y * y <= radius * radius)
                    {
                        GameObject newEntity = SpawnEntity();
                        Vector3 entityPos = new Vector3(hit.point.x + x, hit.point.y, hit.point.z + y);
                        newEntity.transform.position = entityPos;
                        newEntity.transform.localScale = new Vector3(0.1f, 0.1f, 0.1f);
                    }
                }
            }

        }
    }

    void UpdateInfluenceMap()
    {
        // Check if map changed?

        if(InfluenceMap.Instance.useNativeMap)
        {
            if (influenceMap.IsCreated)
            {
                influenceMap.Dispose();
            }

            influenceMap = new NativeArray<float>(InfluenceMap.Instance.influenceMapNativePrev, Allocator.Persistent);
        }
        else
        {
            if (influenceMap.IsCreated)
            {
                influenceMap.Dispose();
            }

            influenceMap = new NativeArray<float>(InfluenceMap.Instance.influenceMapPrev, Allocator.Persistent);
        }

        
        //InfluenceMap.Instance.influenceMapNativePrev.CopyTo(influenceMap);
    }

    void ClearEntities()
    {
        if (Input.GetKeyDown(KeyCode.Backspace))
        {
            for (int i = 0; i < entityList.Count; i++)
            {
                Destroy(entityList[i]);
            }

            entitySidesList.Clear();
            entityList.Clear();
            entityTransformList.Clear();

            movementJobHandle.Complete();

            if (entitySidesArray.IsCreated)
            {
                entitySidesArray.Dispose();
            }
            
            if(transformAccessArray.isCreated)
            {
                transformAccessArray.Dispose();
            }
        }
    }

    private void OnDestroy()
    {
        transformAccessArray.Dispose();
        entitySidesArray.Dispose();
        influenceMap.Dispose();
        offsetArray.Dispose();
    }
}
