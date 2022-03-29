using System.Collections;
using System.Collections.Generic;
using System.Threading;
using UnityEngine;
using Diagnostics = System.Diagnostics;
using Unity.Jobs;
using Unity.Collections;
using Unity.Burst;
using Unity.Burst.Intrinsics;
using Unity.Collections.LowLevel.Unsafe;
using TMPro;

using static Unity.Burst.Intrinsics.X86;
using static Unity.Burst.Intrinsics.X86.Sse;
using static Unity.Burst.Intrinsics.X86.Sse2;
using static Unity.Burst.Intrinsics.X86.Sse3;
using static Unity.Burst.Intrinsics.X86.Ssse3;
using static Unity.Burst.Intrinsics.X86.Sse4_1;
using static Unity.Burst.Intrinsics.X86.Sse4_2;
using static Unity.Burst.Intrinsics.X86.Popcnt;
using static Unity.Burst.Intrinsics.X86.Avx;
using static Unity.Burst.Intrinsics.X86.Avx2;
using static Unity.Burst.Intrinsics.X86.Fma;


public class Timer
{
    Diagnostics.Stopwatch stopwatch;
    double elapsedMs = 0;
    string timerName = "";

    public Timer()
    {
        stopwatch = new Diagnostics.Stopwatch();
        stopwatch.Start();
    }

    public double GetTime()
    {
        // Tick is 100ns
        elapsedMs = stopwatch.ElapsedTicks / 10000.0f;
        return elapsedMs;
    }

    public string GetTimeStr()
    {
        // Tick is 100ns
        elapsedMs = stopwatch.ElapsedTicks / 10000.0f;
        return elapsedMs + "ms";
    }

    public void StartTimer()
    {
        // Tick is 100ns
        stopwatch.Start();
    }

    public void RestartTimer()
    {
        stopwatch.Restart();
    }

    public void StopTimer()
    {
        // Tick is 100ns
        stopwatch.Stop();
    }

    public void ResetTimer()
    {
        // Tick is 100ns
        stopwatch.Reset();
    }

    public void PrintTime()
    {
        // Tick is 100ns
        elapsedMs = stopwatch.ElapsedTicks / 10000.0f;
        Debug.Log(elapsedMs + "ms");
    }

    public void PrintTime(string timerName)
    {
        // Tick is 100ns
        elapsedMs = stopwatch.ElapsedTicks / 10000.0f;
        Debug.Log(timerName + ": " + elapsedMs + "ms");
    }

}

struct Unit
{
    public int x;
    public int y;
    public float influence;
}

public class InfluenceMap : MonoBehaviour
{
    public static InfluenceMap Instance;

    [BurstCompile(FloatPrecision.Standard, FloatMode.Fast)]
    struct InfluenceMapJob : IJobParallelFor
    {
        [ReadOnly] public NativeArray<float> unitMap;
        public NativeArray<float> influenceMap;

        [ReadOnly] public NativeArray<int> unitXCords;
        [ReadOnly] public NativeArray<int> unitYCords;
        [ReadOnly] public NativeArray<float> unitInfs;

        [ReadOnly] public NativeArray<Unit> units;


        const int _CMP_NEQ_OQ = 0x0C;

        //Calculates Euclidean distance between points
        float Distance(int col0, int row0, int col1, int row1)
        {
            return Mathf.Sqrt((col1 - col0) * (col1 - col0) + (row1 - row0) * (row1 - row0));
        }

        // SOME INTRINSICS REQUIRE TO ENABLED IN PROJECT SETTINGS 
        float horizontal_sumf(v256 x)
        {
            v128 hi = mm256_extractf128_ps(x, 1);
            v128 lo = mm256_extractf128_ps(x, 0);
            lo = add_ps(hi, lo);
            hi = movehl_ps(hi, lo);
            lo = add_ps(hi, lo);
            hi = shuffle_ps(lo, lo, 1);
            lo = add_ss(hi, lo);
            return cvtss_f32(lo);
        }

        v256 CalcDistVector8f_v256(v256 c0, v256 r0, v256 c1, v256 r1)
        {
            if (IsFmaSupported)
            {
                v256 colSubVector = mm256_sub_ps(c1, c0);

                v256 rowSubVector = mm256_sub_ps(r1, r0);

                v256 rowMulVector = mm256_mul_ps(rowSubVector, rowSubVector);

                v256 sumVector = mm256_fmadd_ps(colSubVector, colSubVector, rowMulVector);

                v256 sqrtVector = mm256_sqrt_ps(sumVector);

                return sqrtVector;
            }
            else
            {
                v256 colSubVector = mm256_sub_ps(c1, c0);

                v256 rowSubVector = mm256_sub_ps(r1, r0);

                v256 colMulVector = mm256_mul_ps(colSubVector, colSubVector);
                v256 rowMulVector = mm256_mul_ps(rowSubVector, rowSubVector);

                v256 sumVector = mm256_add_ps(colMulVector, rowMulVector);


                v256 rsqrtVector = mm256_rsqrt_ps(sumVector);
                v256 sqrtVector = mm256_rcp_ps(rsqrtVector); // minimal loss of accuracy (max relative error: 1.5*2^-12)

                return sqrtVector;
            }
        }

        //Requires allow unsafe code in player settings
        unsafe float CalcPointInfAVX(int col, int row)
        {
            float inf = 0;
            const int end = ROWS - 8;
            const int rest = ROWS % 8;

            v256 iVec = mm256_set_ps(0, 1, 2, 3, 4, 5, 6, 7);
            v256 resVector = mm256_set1_ps(0);
            v256 addVector = mm256_set1_ps(1.0f);
            v256 col0Vector = mm256_set1_ps(col);
            v256 row0Vector = mm256_set1_ps(row);

            for (int x = 0; x < COLS; x++)
            {
                v256 col1Vector = mm256_set1_ps(x);

                for (int y = 0; y < end; y += 8)
                {
                    v256 row1Vector = mm256_add_ps(iVec, mm256_set1_ps(y));

                    v256 distsVec = CalcDistVector8f_v256(col0Vector, row0Vector, col1Vector, row1Vector);

                    // Add 1 to dists per the formula inf = infl / (1 + dist);
                    distsVec = mm256_add_ps(distsVec, addVector); // add 1 to all values

                    int index = x * ROWS + y;
                    //v256 infsVec = mm256_set_ps((float)unitMap[index + 0], (float)unitMap[index + 1], (float)unitMap[index + 2], (float)unitMap[index + 3], (float)unitMap[index + 4], (float)unitMap[index + 5], (float)unitMap[index + 6], (float)unitMap[index + 7]);

                    //v256 infsVec = mm256_loadu_ps(&unitMap.ReinterpretLoad(index));
                    //UnsafeUtility.AddressOf<float>(unitMap.GetUnsafePtr(), index);
                    //void* ptr = NativeArrayUnsafeUtility.GetUnsafePtr((float*)&unitMap[index]);

                    // Causes burst error in editor because of [ReadOnly], but is faster
                    var ptr = (float*)NativeArrayUnsafeUtility.GetUnsafePtr(unitMap);
                    var offset = ptr + index;
                    v256 infsVec = mm256_loadu_ps(offset);

                    // Check for non zero vectors
                    v256 vcmp = mm256_cmp_ps(infsVec, mm256_set1_ps(0.0f), _CMP_NEQ_OQ); // Element in vcmp is set to all 0's if corresponding element in infsVec is 0, else corresponding element in vcmp is set to all 1's
                    int mask = mm256_movemask_ps(vcmp); // First 8 bits of mask correspond to the sign bits of elements in vcmp 
                    bool anyNonZero = mask != 0; // True if any of masks first 8 bits is 1(at least 1 non zero element)


                    //resVector = _mm256_add_ps(resVector, _mm256_div_ps(infsVec, distsVec));
                    if (anyNonZero)
                    {
                        resVector = mm256_add_ps(resVector, mm256_mul_ps(infsVec, mm256_rcp_ps(distsVec)));
                    }
                }

                // Calculate the rest normally. For example(300 % 8) = 4 (~10% of the total) (Worst case COLS % 8 == 7 which makes this scalar loop almost 20% of the total = BAD)
                for (int y = 0; y < rest; y++) // Try to get rid of this somehow(Find a general solution)
                {
                    float dist = Distance(col, row, x, end + y);
                    int index = x * ROWS + y;
                    inf += unitMap[index] / (1.0f + dist);
                }
            }

            inf += horizontal_sumf(resVector);
            return inf;
        }

        float CalcPointInf(int col, int row)
        {
            float totalInf = 0.0f;
            for (int i = 0; i < unitInfs.Length; i++)
            {
                Unity.Burst.CompilerServices.Loop.ExpectVectorized();
                int x = unitXCords[i];
                int y = unitYCords[i];

                float dist = Distance(col, row, x, y);

                totalInf += unitInfs[i] / (1 + dist);
            }
            return totalInf;
        }

        float CalcPointInfStruct(int col, int row)
        {
            float totalInf = 0.0f;
            for (int i = 0; i < units.Length; i++)
            {
                Unity.Burst.CompilerServices.Loop.ExpectVectorized();
                int x = units[i].x;
                int y = units[i].y;

                float dist = Distance(col, row, x, y);

                totalInf += units[i].influence / (1 + dist);
            }
            return totalInf;
        }

        unsafe float CalcPointInfAVX_2(int col, int row)
        {
            float totalInf = 0.0f;
            int unitCount = unitInfs.Length;
            int end = unitCount - 8;
            int rest = unitCount % 8;

            //v256 iVector = mm256_set_ps(0, 1, 2, 3, 4, 5, 6, 7);
            v256 resVector = mm256_set1_ps(0);
            v256 addVector = mm256_set1_ps(1.0f); // Distance dropoff factor(smaller is faster dropoff)
            v256 col0Vector = mm256_set1_ps(col);
            v256 row0Vector = mm256_set1_ps(row);

            //int* xCordsPtr = (int*)NativeArrayUnsafeUtility.GetUnsafePtr(unitXCords);
            //int* yCordsPtr = (int*)NativeArrayUnsafeUtility.GetUnsafePtr(unitYCords);
            //int* infsPtr = (int*)NativeArrayUnsafeUtility.GetUnsafePtr(unitInfs);

            for (int i = 0; i < end; i += 8)
            {
                //int* xCordsPtr = (int*)NativeArrayUnsafeUtility.GetUnsafePtr(unitXCords);
                //int* xCordsOffset = xCordsPtr + i;
                ////v256 xVector = mm256_lddqu_si256(xCordsOffset);
                ////v256 xVector = mm256_loadu_si256(xCordsOffset);
                //v256 xVector = mm256_cvtepi32_ps(mm256_loadu_si256(xCordsOffset));

                v256 xVector = mm256_set_ps(unitXCords[i + 0], unitXCords[i + 1], unitXCords[i + 2], unitXCords[i + 3],
                    unitXCords[i + 4], unitXCords[i + 5], unitXCords[i + 6], unitXCords[i + 7]);

                //int* yCordsPtr = (int*)NativeArrayUnsafeUtility.GetUnsafePtr(unitYCords);
                //int* yCordsOffset = yCordsPtr + i;
                ////v256 yVector = mm256_lddqu_si256(yCordsOffset);
                ////v256 yVector = mm256_loadu_si256(yCordsOffset);
                //v256 yVector = mm256_cvtepi32_ps(mm256_loadu_si256(yCordsOffset));

                v256 yVector = mm256_set_ps(unitYCords[i + 0], unitYCords[i + 1], unitYCords[i + 2], unitYCords[i + 3],
                    unitYCords[i + 4], unitYCords[i + 5], unitYCords[i + 6], unitYCords[i + 7]);

                v256 distVector = CalcDistVector8f_v256(col0Vector, row0Vector, xVector, yVector);

                distVector = mm256_add_ps(distVector, addVector);

                //float* infsPtr = (float*)NativeArrayUnsafeUtility.GetUnsafePtr(unitInfs);
                //float* infsOffset = infsPtr + i;
                //v256 infVector = mm256_loadu_ps(infsOffset);

                v256 infVector = mm256_set_ps(unitInfs[i + 0], unitInfs[i + 1], unitInfs[i + 2], unitInfs[i + 3],
                    unitInfs[i + 4], unitInfs[i + 5], unitInfs[i + 6], unitInfs[i + 7]);

                v256 vcmp = mm256_cmp_ps(infVector, mm256_set1_ps(0.0f), _CMP_NEQ_OQ); // Element in vcmp is set to all 0's if corresponding element in infsVec is 0, else corresponding element in vcmp is set to all 1's
                int mask = mm256_movemask_ps(vcmp); // First 8 bits of mask correspond to the sign bits of elements in vcmp 
                bool anyNonZero = mask != 0; // True if any of masks first 8 bits is 1(at least 1 non zero element)

                //resVector = _mm256_add_ps(resVector, _mm256_div_ps(infsVec, distsVec));
                if (anyNonZero)
                {
                    resVector = mm256_add_ps(resVector, mm256_mul_ps(infVector, mm256_rcp_ps(distVector)));
                }
            }

            totalInf += horizontal_sumf(resVector);

            // Calculate the rest normally. For example(300 % 8) = 4 (~10% of the total) (Worst case COLS % 8 == 7 which makes this scalar loop almost 20% of the total = BAD)
            for (int i = 0; i < rest; i++) // Try to get rid of this somehow(Find a general solution)
            {
                int x = unitXCords[i];
                int y = unitYCords[i];
                float dist = Distance(col, row, x, y);
                totalInf += unitInfs[i] / (1.0f + dist);
            }

            return totalInf;
        }


        

        // Calculates and returns influence for a single point in the influence map
        float CalculatePointInfluence(int col, int row)
        {
            float totalInfluence = 0.0f;
            for (int x = 0; x < COLS; x++)
            {
                for (int y = 0; y < ROWS; y++)
                {
                    float distance = Distance(col, row, x, y);
                    int index = x * ROWS + y;
                    totalInfluence += unitMap[index] / (1 + distance); // Influence of units on a point drops with relative to distance
                }
            }

            return totalInfluence;
        }

        public void Execute(int index)
        {
            //if (IsAvxSupported)
            //{
            //    int x = index / ROWS;
            //    int y = index % ROWS;

            //    influenceMap[index] = CalcPointInfAVX(x, y);
            //}
            //else
            //{
            //    int x = index / ROWS;
            //    int y = index % ROWS;

            //    //float influencePoint = CalculatePointInfluence(x, y);
            //    float influencePoint = CalcPointInf(x, y);
            //    influenceMap[index] = influencePoint;
            //}

            int x = index / ROWS;
            int y = index % ROWS;

            //float influencePoint = CalculatePointInfluence(x, y);
            //float influencePoint = CalcPointInf(x, y);
            float influencePoint = CalcPointInfStruct(x, y);
            //float influencePoint = CalcPointInfAVX_2(x, y);
            influenceMap[index] = influencePoint;

            //int x = index / ROWS;
            //int y = index % ROWS;

            //float influencePoint = CalculatePointInfluence(x, y);
            //influenceMap[index] = influencePoint;


            //float influencePoint = CalculatePointInfluence(x, y);

        }

        //public void Execute(int index)
        //{
        //    int x = index / ROWS;
        //    int y = index % ROWS - 1;
            
        //    float influencePoint = CalculatePointInfluence(x, y);
        //    influenceMap[index] = influencePoint;
        //}
    }


    [SerializeField] GameObject influenceMapObj;
    [SerializeField] Renderer renderer;
    Texture2D influenceMapTexture;

    [SerializeField] GameObject entity;

    const int ROWS = 320;
    const int COLS = 320;

    const int minUnitsPerSide = 10;
    const int maxUnitsPerSide = 20;

    const float minInfluence = 1.0f;
    const float maxInfluence = 5.0f;

    int[,] offsets = new int[,] { { 0, 1 }, { 1, 1 }, { 1, 0 }, { 1, -1 }, { 0, -1 }, { -1, -1 }, { -1, 0 }, { -1, 1 } };
    float entityMoveSpeed = 0.1f;


    float[,] unitsMap = new float[COLS, ROWS];
    float[,] influenceMap = new float[COLS, ROWS];
    float[,] influenceMapPrev = new float[COLS, ROWS];

    public NativeArray<int> unitXCords;
    public NativeArray<int> unitYCords;
    public NativeArray<float> unitInfs;


    public NativeArray<float> unitMapNative;
    public NativeArray<float> influenceMapNative;
    public NativeArray<float> influenceMapNativePrev;


    Color32 redColor = new Color32(255, 0, 0, 255);
    Color32 greenColor = new Color32(0, 255, 0, 255);

    int textureID;
    int textureColorID;

    Thread computeThread = null;
    bool computeThreadJoined = true;
    bool computeReady = true;

    JobHandle influenceMapJobHandle;
    InfluenceMapJob influenceMapJob;

    bool influenceJobReady = false;

    Timer jobTimer;

    [SerializeField] TMP_Text timerText;
    [SerializeField] TMP_Text mapSizeText;
    [SerializeField] TMP_Text entityCountText;


    [SerializeField] GameObject entityPrefab;

    List<GameObject> entityList;

    const int entityCount = 100;


    private void Awake()
    {
        if(Instance != null && Instance != this)
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
        //influenceMapObj.transform.localScale = new Vector3(COLS, ROWS, ROWS);
        unitMapNative = new NativeArray<float>(ROWS* COLS, Allocator.Persistent);
        influenceMapNative = new NativeArray<float>(ROWS* COLS, Allocator.Persistent);
        influenceMapNativePrev = new NativeArray<float>(ROWS* COLS, Allocator.Persistent);

        influenceMapTexture = new Texture2D(ROWS, COLS, TextureFormat.RGBA32, false);
        influenceMapTexture.filterMode = FilterMode.Point;
        textureID = Shader.PropertyToID("_MainTex");
        textureColorID = Shader.PropertyToID("_Color");

        GenerateRandomNativeMap(unitMapNative);

        unitXCords = new NativeArray<int>(10, Allocator.Persistent);
        unitYCords = new NativeArray<int>(10, Allocator.Persistent);
        unitInfs = new NativeArray<float>(10, Allocator.Persistent);
        GenerateRandomUnits();

        //GenerateRandomNativeMapFull(unitMapNative);

        mapSizeText.text = "Map size: " + COLS + " x " + ROWS;
        jobTimer = new Timer();

        //for (int i = -3; i < 4; i++)
        //{
        //    for (int j = -3; j < 4; j++)
        //    {
        //        offsets[i, j] = i
        //    }
        //}
        offsets = new int[,] { { 0, 1 }, { 1, 1 }, { 1, 0 }, { 1, -1 }, { 0, -1 }, { -1, -1 }, { -1, 0 }, { -1, 1 } };


        entityList = new List<GameObject>();

        //for (int i = 0; i < entityCount; i++)
        //{
        //    GameObject newEntity = Instantiate(entityPrefab);
        //    entityList.Add(newEntity);
        //    entityScriptList.Add(newEntity.GetComponent<EntityScript>());
        //    Vector3 entityPos = new Vector3((COLS / 2) - (Mathf.Sqrt(entityCount) / 2) + i % (int)Mathf.Sqrt(entityCount), 0.0f, (ROWS / 2) - (Mathf.Sqrt(entityCount) / 2) + i % (int)Mathf.Sqrt(entityCount));
        //    newEntity.transform.position = entityPos;
        //}

        //offsets = new int[,] { { 0, 3 }, { 3, 3 }, { 3, 0 }, { 3, -3 }, { 0, -3 }, { -3, -3 }, { -3, 0 }, { -3, 3 } };
        //GenerateRandomMap(unitsMap);
        //GenerateRandomMapFull(unitsMap);
        //computeThread = new Thread(CalculateInfluenceMap);
        //computeThread.Name = "ComputeThread";
        //computeThreadJoined = false;
        //computeReady = false;
        //computeThread.Start();
    }

    


    // Update is called once per frame
    void Update()
    {
        entityCountText.text = "Entities: " + entityList.Count;

        if (!computeThreadJoined && computeReady)
        {
            computeThread.Join();
            computeThreadJoined = true;
            jobTimer.PrintTime("Influence time: ");
            timerText.text = "Time:" + jobTimer.GetTimeStr();
            RenderMapToTexture(influenceMapTexture);
            Debug.Log("Compute thread joined!");
        }

        //if (computeThreadJoined && Input.GetKeyDown(KeyCode.LeftControl))
        //{
        //    StartComputeThread();
        //}

        if (computeThreadJoined && Input.GetKeyDown(KeyCode.Return))
        {
            GenerateRandomUnits();

            GenerateRandomNativeMap(unitMapNative);
            //GenerateRandomNativeMapFull(unitMapNative);
            Debug.Log("Starting a new influenceMapJob");
            influenceMapJob = new InfluenceMapJob();
            influenceMapJob.influenceMap = influenceMapNative;
            influenceMapJob.unitMap = unitMapNative;
            influenceMapJob.unitXCords = unitXCords;
            influenceMapJob.unitYCords = unitYCords;
            influenceMapJob.unitInfs = unitInfs;

            jobTimer.RestartTimer();
            influenceMapJobHandle = influenceMapJob.Schedule(ROWS * COLS, ROWS);
            //JobHandle.ScheduleBatchedJobs();
            influenceJobReady = false;

            influenceMapJobHandle.Complete();
            jobTimer.StopTimer();
            jobTimer.PrintTime("Influence job time: ");
            timerText.text = "Influence map time: " + jobTimer.GetTimeStr();
            Debug.Log("InfluenceMapJob completed!");
            //influenceMapNativePrev = influenceMapNative;
            influenceMapNative.CopyTo(influenceMapNativePrev);

            //EntityManager.Instance.UpdateInfluenceMap(); // Dangerous to call from here

            RenderNativeMapToTexture(influenceMapTexture);
            influenceJobReady = true;

        }


        if(!influenceJobReady)
        {
            if(influenceMapJobHandle.IsCompleted)
            {
                //influenceMapJobHandle.Complete();
                ////unitXCords.Dispose();
                ////unitYCords.Dispose();
                ////unitInfs.Dispose();
                //jobTimer.PrintTime("Influence job time: ");
                //timerText.text = "Time:" + jobTimer.GetTimeStr();
                //Debug.Log("InfluenceMapJob completed!");
                ////influenceMapNativePrev = influenceMapNative;
                //influenceMapNative.CopyTo(influenceMapNativePrev);
                //RenderNativeMapToTexture(influenceMapTexture);
                //influenceJobReady = true;
            }
            
            
            //Debug.Log(influenceMapNative.Length);
            //if (influenceMapNative.Length > 0)
            //{
            //    RenderNativeMapToTexture(influenceMapTexture);
            //}

            //influenceMapNativePrev = influenceMapNative;
            
            
        }

        //if(influenceMapNativePrev.IsCreated && influenceMapNativePrev.Length > 0)
        //{
        //    //MoveEntity();
        //    if(entityList.Count > 0)
        //    {
        //        MoveEntities();
        //    }
            
        //}
        

        
        //RenderMapToTexture(influenceMapTexture);
    }

    private void OnDestroy()
    {
        unitMapNative.Dispose();
        influenceMapNative.Dispose();
        influenceMapNativePrev.Dispose();

        unitXCords.Dispose();
        unitYCords.Dispose();
        unitInfs.Dispose();
    }

    public int GetMapRows()
    {
        return ROWS;
    }

    public int GetMapCols()
    {
        return COLS;
    }

    //private void OnDisable()
    //{
    //    unitMapNative.Dispose();
    //    influenceMapNative.Dispose();
    //    influenceMapNativePrev.Dispose();
    //}

    public void StartComputeThread()
    {
        jobTimer.RestartTimer();
        GenerateRandomMap(unitsMap);
        //GenerateRandomMapFull(unitsMap);
        computeThread = new Thread(CalculateInfluenceMap);
        computeThread.Name = "ComputeThread";
        computeThreadJoined = false;
        computeReady = false;
        computeThread.Start();
    }


    void GenerateRandomUnits()
    {
        Debug.Log("Generating random units...");

        int unitsGreen = Random.Range(minUnitsPerSide, maxUnitsPerSide);
        int unitsRed = Random.Range(minUnitsPerSide, maxUnitsPerSide);
        
        int totalUnits = unitsGreen + unitsRed;

        NativeArray<int> xCords = new NativeArray<int>(totalUnits, Allocator.Persistent);
        NativeArray<int> yCords = new NativeArray<int>(totalUnits, Allocator.Persistent);
        NativeArray<float> infs = new NativeArray<float>(totalUnits, Allocator.Persistent);

        influenceMapJobHandle.Complete();

        unitXCords.Dispose();
        unitYCords.Dispose();
        unitInfs.Dispose();

        unitXCords = xCords;
        unitYCords = yCords;
        unitInfs = infs;


        for (int i = 0; i < unitsGreen; i++)
        {
            int x = Random.Range(0, COLS - 1);
            int y = Random.Range(0, ROWS - 1);

            xCords[i] = x;
            yCords[i] = y;
            infs[i] = Random.Range(minInfluence, maxInfluence);
        }


        for (int i = unitsGreen; i < totalUnits; i++)
        {
            int x = Random.Range(0, COLS - 1);
            int y = Random.Range(0, ROWS - 1);

            xCords[i] = x;
            yCords[i] = y;
            infs[i] = Random.Range(minInfluence, maxInfluence) * -1.0f;
        }

        Debug.Log("Random units generated!");
    }

    void GenerateRandomNativeMap(NativeArray<float> map)
    {
        Debug.Log("Generating random map...");
        ResetNativeMap(map);

        int unitsRed = Random.Range(minUnitsPerSide, maxUnitsPerSide);
        int unitsGreen = Random.Range(minUnitsPerSide, maxUnitsPerSide);

        for (int i = 0; i < unitsRed; i++)
        {
            int x = Random.Range(0, COLS - 1);
            int y = Random.Range(0, ROWS - 1);

            int index = x * ROWS + y;
            map[index] = Random.Range(minInfluence, maxInfluence);
        }


        for (int i = 0; i < unitsGreen; i++)
        {
            int x = Random.Range(0, COLS - 1);
            int y = Random.Range(0, ROWS - 1);

            int index = x * ROWS + y;
            map[index] = Random.Range(minInfluence, maxInfluence) * -1.0f;
        }

        Debug.Log("Random native map generated!");
    }
    void GenerateRandomMap(float[,] map)
    {
        Debug.Log("Generating random map...");
        ResetMap(map);

        int unitsRed = Random.Range(minUnitsPerSide, maxUnitsPerSide);
        int unitsGreen = Random.Range(minUnitsPerSide, maxUnitsPerSide);

        for (int i = 0; i < unitsRed; i++)
        {
            int x = Random.Range(0, COLS - 1);
            int y = Random.Range(0, ROWS - 1);

            map[x, y] = Random.Range(minInfluence, maxInfluence);
        }


        for (int i = 0; i < unitsGreen; i++)
        {
            int x = Random.Range(0, COLS - 1);
            int y = Random.Range(0, ROWS - 1);

            map[x, y] = Random.Range(minInfluence, maxInfluence) * -1.0f;
        }

        Debug.Log("Random map generated!");
    }

    void GenerateRandomMapFull(float[,] map)
    {
        Debug.Log("Generating random map full...");
        ResetMap(map);

        for (int x = 0; x < COLS; x++)
        {
            for (int y = 0; y < ROWS; y++)
            {
                int side = Random.Range(0, 2);
                map[x, y] = Random.Range(minInfluence, maxInfluence);

                if (side == 1)
                {
                    map[x, y] *= -1.0f;
                }
            }
        }

        Debug.Log("Random map filled with units generated!");
    }

    void GenerateRandomNativeMapFull(NativeArray<float> map)
    {
        Debug.Log("Generating random native map full...");
        ResetNativeMap(map);

        for (int x = 0; x < COLS; x++)
        {
            for (int y = 0; y < ROWS; y++)
            {
                int side = Random.Range(0, 2);
                int index = x * ROWS + y;
                map[index] = Random.Range(minInfluence, maxInfluence);

                if (side == 1)
                {
                    map[index] *= -1.0f;
                }
            }
        }

        Debug.Log("Random native map filled with units generated!");
    }

    void ResetNativeMap(NativeArray<float> map)
    {
        Debug.Log("Resetting map...");
        for (int x = 0; x < COLS; x++)
        {
            for (int y = 0; y < ROWS; y++)
            {
                int index = x * ROWS + y;
                map[index] = 0;
            }
        }
        Debug.Log("Map reset!");
    }

    void ResetMap(float[,] map)
    {
        Debug.Log("Resetting map...");
        for (int x = 0; x < COLS; x++)
        {
            for (int y = 0; y < COLS; y++)
            {
                map[x, y] = 0;
            }
        }
        Debug.Log("Map reset!");
    }

    // Calculates Euclidean distance between points
    float Distance(int col0, int row0, int col1, int row1)
    {
        return Mathf.Sqrt((float)(col1 - col0) * (col1 - col0) + (row1 - row0) * (row1 - row0));
    }

    // Calculates and returns influence for a single point in the influence map
    float CalculatePointInfluence(int col, int row)
    {
        float totalInfluence = 0.0f;
        for (int x = 0; x < COLS; x++)
        {
            for (int y = 0; y < ROWS; y++)
            {
                float distance = Distance(col, row, x, y);
                //Debug.Log("Distance: " + distance);
                totalInfluence += unitsMap[x, y] / (1 + distance); // Influence of units on a point drops with relative to distance
            }
        }

        return totalInfluence;
    }

    void CalculateInfluenceMap()
    {
        Debug.Log(Thread.CurrentThread.Name + " computing...");
        Debug.Log("Calculating influence map...");
        Timer influenceTimer = new Timer();
        for (int x = 0; x < COLS; x++)
        {
            for (int y = 0; y < COLS; y++)
            {
                influenceMap[x, y] = CalculatePointInfluence(x, y);
            }
        }
        influenceTimer.PrintTime("InfluenceTimer");
        Debug.Log("Influence map calculated");
        computeReady = true;
    }

    void RenderMapToTexture(Texture2D mapTexture)
    {
        //Debug.Log("Rendering map to texture...");
        Timer renderTimer = new Timer();
        for (int x = 0; x < COLS; x++)
        {
            for (int y = 0; y < ROWS; y++)
            {
                float alpha = 255 * Mathf.Abs(influenceMap[x, y]);

                if (alpha < 0.0f)
                {
                    alpha = 0.0f;
                }
                else if (alpha > 255.0f)
                {
                    alpha = 255.0f;
                }


                if (influenceMap[x, y] > 0)
                {
                    greenColor.a = (byte)alpha;
                    mapTexture.SetPixel(x, y, greenColor);
                }
                else
                {
                    redColor.a = (byte)alpha;
                    mapTexture.SetPixel(x, y, redColor);
                }
            }
        }

        mapTexture.SetPixel(30, 100, Color.blue);
        mapTexture.Apply();
        renderer.material.SetTexture(textureID, mapTexture);
        renderer.material.SetColor(textureColorID, Color.white);
        //renderTimer.PrintTime("RenderTimer");
        //Debug.Log("Rendered map to texture");
    }

    void RenderNativeMapToTexture(Texture2D mapTexture)
    {
        Timer renderTimer = new Timer();
        for (int x = 0; x < COLS; x++)
        {
            for (int y = 0; y < ROWS; y++)
            {
                int index = x * ROWS + y;

                float influence = influenceMapNative[index];

                float alpha = 255 * Mathf.Abs(influence);

                if (alpha < 0.0f)
                {
                    alpha = 0.0f;
                }
                else if (alpha > 255.0f)
                {
                    alpha = 255.0f;
                }


                if (influence >= 0)
                {
                    greenColor.a = (byte)alpha;
                    mapTexture.SetPixel(x, y, greenColor);
                }
                else
                {
                    redColor.a = (byte)alpha;
                    mapTexture.SetPixel(x, y, redColor);
                }
            }
        }


        mapTexture.Apply();
        renderer.material.SetTexture(textureID, mapTexture);
        renderer.material.SetColor(textureColorID, Color.white);
        //renderTimer.PrintTime("RenderTimer");
        //Debug.Log("Rendered map to texture");
    }
}


