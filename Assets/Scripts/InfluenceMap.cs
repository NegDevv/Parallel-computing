using System.Collections.Generic;
using TMPro;
using Unity.Burst;
using Unity.Burst.Intrinsics;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using UnityEngine;
using static Unity.Burst.Intrinsics.X86.Avx;
using static Unity.Burst.Intrinsics.X86.Fma;
using static Unity.Burst.Intrinsics.X86.Sse;
using static Unity.Mathematics.math;
using Diagnostics = System.Diagnostics;
using float4 = Unity.Mathematics.float4;


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

    public void Start()
    {
        // Tick is 100ns
        stopwatch.Start();
    }

    public void Restart()
    {
        stopwatch.Restart();
    }

    public void Stop()
    {
        // Tick is 100ns
        stopwatch.Stop();
    }

    public void Reset()
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

public struct Unit
{
    public int x;
    public int y;
    public float influence;
}

public class InfluenceMap : MonoBehaviour
{
    public static InfluenceMap Instance;

    const int deterministicSeed = 118355416;

    [BurstCompile(FloatPrecision.Standard, FloatMode.Fast)]
    struct InfluenceMapJobUnitStruct : IJobParallelFor
    {
        public NativeArray<float> influenceMap;

        [ReadOnly] public NativeArray<Unit> units;

        float CalculatePointInfluenceUnitStructFloat4(int col, int row)
        {
            float totalInf = 0.0f;
            int unitCount = units.Length;

            int end = unitCount - 4;
            int rest = unitCount % 4;

            float4 resVector = new float4(0.0f); // Result accumulator
            float4 addVector = new float4(1.0f); // Distance dropoff factor(smaller is faster dropoff)
            float4 col0Vector = new float4(col);
            float4 row0Vector = new float4(row);


            for (int i = 0; i < end; i += 4)
            {
                float4 xVector = new float4(units[i + 0].x, units[i + 1].x, units[i + 2].x, units[i + 3].x);

                float4 yVector = new float4(units[i + 0].y, units[i + 1].y, units[i + 2].y, units[i + 3].y);

                float4 distVector = CalcDistFloat4(col0Vector, row0Vector, xVector, yVector);

                distVector = distVector + addVector;

                float4 infVector = new float4(units[i + 0].influence, units[i + 1].influence, units[i + 2].influence, units[i + 3].influence);

                //resVector = resVector + (infVector / distVector);
                resVector = resVector + (infVector * rcp(distVector));
            }

            // Add horizontal sum of resVector to totalInf
            totalInf += csum(resVector);

            // Calculate the rest normally. 
            for (int i = 0; i < rest; i++)
            {
                int x = units[i].x;
                int y = units[i].y;
                float dist = Distance(col, row, x, y);
                totalInf += units[i].influence / (1.0f + dist);
            }

            return totalInf;
        }

        unsafe float CalcPointInfUnitStructAVX(int col, int row)
        {
            float totalInf = 0.0f;
            int unitCount = units.Length;

            int end = unitCount - 8;
            int rest = unitCount % 8;

            v256 resVector = mm256_set1_ps(0); // Result accumulator
            v256 addVector = mm256_set1_ps(1.0f); // Distance dropoff factor(smaller is faster dropoff)
            v256 col0Vector = mm256_set1_ps(col);
            v256 row0Vector = mm256_set1_ps(row);


            for (int i = 0; i < end; i += 8)
            {

                v256 xVector = mm256_set_ps(units[i + 0].x, units[i + 1].x, units[i + 2].x, units[i + 3].x,
                    units[i + 4].x, units[i + 5].x, units[i + 6].x, units[i + 07].x);


                v256 yVector = mm256_set_ps(units[i + 0].y, units[i + 1].y, units[i + 2].y, units[i + 3].y,
                    units[i + 4].y, units[i + 5].y, units[i + 6].y, units[i + 7].y);

                v256 distVector = CalcDistVector8f_v256(col0Vector, row0Vector, xVector, yVector);

                distVector = mm256_add_ps(distVector, addVector);


                v256 infVector = mm256_set_ps(units[i + 0].influence, units[i + 1].influence, units[i + 2].influence, units[i + 3].influence,
                    units[i + 4].influence, units[i + 5].influence, units[i + 6].influence, units[i + 7].influence);

                //resVector = mm256_add_ps(resVector, mm256_div_ps(infVector, distVector));
                resVector = mm256_add_ps(resVector, mm256_mul_ps(infVector, mm256_rcp_ps(distVector)));
            }

            totalInf += horizontal_sumf(resVector);

            // Calculate the rest normally. For example(300 % 8) = 4 (~10% of the total) (Worst case COLS % 8 == 7 which makes this scalar loop almost 20% of the total = BAD)
            for (int i = 0; i < rest; i++) // Try to get rid of this somehow(Find a general solution)
            {
                int x = units[i].x;
                int y = units[i].y;
                float dist = Distance(col, row, x, y);
                totalInf += units[i].influence / (1.0f + dist);
            }

            return totalInf;
        }

        float CalcPointInfUnitStruct(int col, int row)
        {
            float totalInf = 0.0f;
            for (int i = 0; i < units.Length; i++)
            {
                //Unity.Burst.CompilerServices.Loop.ExpectVectorized();
                int x = units[i].x;
                int y = units[i].y;

                float dist = Distance(col, row, x, y);

                totalInf += units[i].influence / (1 + dist);
            }
            return totalInf;
        }

        public void Execute(int index)
        {
            if (IsAvxSupported)
            {
                int x = index / ROWS;
                int y = index % ROWS;

                //float influencePoint = CalcPointInfUnitStructAVX(x, y);
                float influencePoint = CalculatePointInfluenceUnitStructFloat4(x, y);
                influenceMap[index] = influencePoint;
            }
            else
            {
                int x = index / ROWS;
                int y = index % ROWS;

                float influencePoint = CalcPointInfUnitStruct(x, y);
                influenceMap[index] = influencePoint;
            }
        }
    }

    [BurstCompile(FloatPrecision.Standard, FloatMode.Fast)]
    struct InfluenceMapJobVectorized : IJobParallelFor
    {
        public NativeArray<float> influenceMap;

        [ReadOnly] public NativeArray<int> unitXCords;
        [ReadOnly] public NativeArray<int> unitYCords;
        [ReadOnly] public NativeArray<float> unitInfs;


        float CalculatePointInfluenceFloat4(int col, int row)
        {
            float totalInf = 0.0f;
            int unitCount = unitInfs.Length;

            int end = unitCount - 4;
            int rest = unitCount % 4;

            float4 resVector = new float4(0.0f); // Result accumulator
            float4 addVector = new float4(1.0f); // Distance dropoff factor(smaller is faster dropoff)
            float4 col0Vector = new float4(col);
            float4 row0Vector = new float4(row);


            for (int i = 0; i < end; i += 4)
            {
                float4 xVector = new float4(unitXCords[i + 0], unitXCords[i + 1], unitXCords[i + 2], unitXCords[i + 3]);

                float4 yVector = new float4(unitYCords[i + 0], unitYCords[i + 1], unitYCords[i + 2], unitYCords[i + 3]);

                float4 distVector = CalcDistFloat4(col0Vector, row0Vector, xVector, yVector);

                distVector = distVector + addVector;

                float4 infVector = new float4(unitInfs[i + 0], unitInfs[i + 1], unitInfs[i + 2], unitInfs[i + 3]);

                //resVector = resVector + (infVector / distVector);
                resVector = resVector + (infVector * rcp(distVector));
            }

            // Add horizontal sum of resVector to totalInf
            totalInf += csum(resVector);

            // Calculate the rest normally. 
            for (int i = 0; i < rest; i++)
            {
                int x = unitXCords[i];
                int y = unitYCords[i];
                float dist = Distance(col, row, x, y);
                totalInf += unitInfs[i] / (1.0f + dist);
            }

            return totalInf;
        }


        // Requires allow unsafe code in player settings
        unsafe float CalcPointInfUnsafeAVX(int col, int row)
        {
            float totalInf = 0.0f;
            int unitCount = unitInfs.Length;

            int end = unitCount - 8;
            int rest = unitCount % 8;

            v256 resVector = mm256_set1_ps(0); // Result accumulator
            v256 addVector = mm256_set1_ps(1.0f); // Distance dropoff factor(smaller is faster dropoff)
            v256 col0Vector = mm256_set1_ps(col);
            v256 row0Vector = mm256_set1_ps(row);

            int* xCordsPtr = (int*)NativeArrayUnsafeUtility.GetUnsafePtr(unitXCords);
            int* yCordsPtr = (int*)NativeArrayUnsafeUtility.GetUnsafePtr(unitYCords);
            float* infsPtr = (float*)NativeArrayUnsafeUtility.GetUnsafePtr(unitInfs);

            for (int i = 0; i < end; i += 8)
            {
                int* xCordsOffset = xCordsPtr + i;
                //v256 xVector = mm256_lddqu_si256(xCordsOffset);
                //v256 xVector = mm256_loadu_si256(xCordsOffset);
                v256 xVector = mm256_cvtepi32_ps(mm256_loadu_si256(xCordsOffset));

                int* yCordsOffset = yCordsPtr + i;
                //v256 yVector = mm256_lddqu_si256(yCordsOffset);
                //v256 yVector = mm256_loadu_si256(yCordsOffset);
                v256 yVector = mm256_cvtepi32_ps(mm256_loadu_si256(yCordsOffset));


                v256 distVector = CalcDistVector8f_v256(col0Vector, row0Vector, xVector, yVector);

                distVector = mm256_add_ps(distVector, addVector);

                float* infsOffset = infsPtr + i;
                v256 infVector = mm256_loadu_ps(infsOffset);

                //resVector = mm256_add_ps(resVector, mm256_div_ps(infVector, distVector));
                resVector = mm256_add_ps(resVector, mm256_mul_ps(infVector, mm256_rcp_ps(distVector)));
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

        float CalcPointInfAVX(int col, int row)
        {
            float totalInf = 0.0f;
            int unitCount = unitInfs.Length;

            int end = unitCount - 8;
            int rest = unitCount % 8;

            v256 resVector = mm256_set1_ps(0); // Result accumulator
            v256 addVector = mm256_set1_ps(1.0f); // Distance dropoff factor(smaller is faster dropoff)
            v256 col0Vector = mm256_set1_ps(col);
            v256 row0Vector = mm256_set1_ps(row);

            for (int i = 0; i < end; i += 8)
            {
                v256 xVector = mm256_set_ps(unitXCords[i + 0], unitXCords[i + 1], unitXCords[i + 2], unitXCords[i + 3],
                    unitXCords[i + 4], unitXCords[i + 5], unitXCords[i + 6], unitXCords[i + 7]);

                v256 yVector = mm256_set_ps(unitYCords[i + 0], unitYCords[i + 1], unitYCords[i + 2], unitYCords[i + 3],
                    unitYCords[i + 4], unitYCords[i + 5], unitYCords[i + 6], unitYCords[i + 7]);

                v256 distVector = CalcDistVector8f_v256(col0Vector, row0Vector, xVector, yVector);

                distVector = mm256_add_ps(distVector, addVector);

                v256 infVector = mm256_set_ps(unitInfs[i + 0], unitInfs[i + 1], unitInfs[i + 2], unitInfs[i + 3],
                    unitInfs[i + 4], unitInfs[i + 5], unitInfs[i + 6], unitInfs[i + 7]);

                //resVector = mm256_add_ps(resVector, mm256_div_ps(infVector, distVector));
                resVector = mm256_add_ps(resVector, mm256_mul_ps(infVector, mm256_rcp_ps(distVector)));
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
            for (int i = 0; i < unitInfs.Length; i++)
            {
                int x = unitXCords[i];
                int y = unitYCords[i];

                float dist = Distance(col, row, x, y);

                totalInfluence += unitInfs[i] / (1 + dist);
            }
            return totalInfluence;
        }

        public void Execute(int index)
        {

            if (IsAvxSupported)
            {
                int x = index / ROWS;
                int y = index % ROWS;

                //float influencePoint = CalcPointInfAVX(x, y);
                //float influencePoint = CalcPointInfUnsafeAVX(x, y);
                float influencePoint = CalculatePointInfluenceFloat4(x, y);
                influenceMap[index] = influencePoint;
            }
            else
            {
                int x = index / ROWS;
                int y = index % ROWS;

                float influencePoint = CalculatePointInfluence(x, y);
                influenceMap[index] = influencePoint;
            }
        }
    }

    [BurstCompile(FloatPrecision.Standard, FloatMode.Fast)]
    struct InfluenceMapJob : IJobParallelFor
    {
        // Vaikutuskartta tallennetaan NativeArray-taulukkoon.
        public NativeArray<float> influenceMap;

        // Yksikködata välitetään kolmena NativeArray-taulukkona.
        [ReadOnly] public NativeArray<int> unitXCords;
        [ReadOnly] public NativeArray<int> unitYCords;
        [ReadOnly] public NativeArray<float> unitInfs;

        // Laskee ja palauttaa vaikutusarvon yhdelle kartan ruudulle.
        float CalculatePointInfluence(int col, int row)
        {
            float totalInfluence = 0.0f; // Kokonaisvaikutus.

            // Käydään läpi kaikki yksiköt ja summataan niiden vaikutus.
            for (int i = 0; i < unitInfs.Length; i++)
            {
                int x = unitXCords[i];
                int y = unitYCords[i];

                // Lasketaan yksikön etäisyys ruudusta.
                float dist = Distance(col, row, x, y);

                // Yksikön vaikutus ruutuun laskee etäisyyden mukaan.
                totalInfluence += unitInfs[i] / (1 + dist);
            }
            return totalInfluence;
        }

        // Kutsutaan yhtä vaikutuskartan ruutua kohden.
        public void Execute(int index)
        {
            // Muunnetaan yksiulotteinen taulukon indeksi x- ja y-koordinaateiksi.
            int x = index / ROWS;
            int y = index % ROWS;

            // Lasketaan ruudun vaikutus ja tallennetaan se vaikutuskarttaan käsiteltävään indeksiin.
            float influencePoint = CalculatePointInfluence(x, y);
            influenceMap[index] = influencePoint;
        }
    }


    [BurstCompile(FloatPrecision.Standard, FloatMode.Fast)]
    struct InfluenceMapJobST : IJob
    {
        // Vaikutuskartta tallennetaan NativeArray-taulukkoon.
        public NativeArray<float> influenceMap;

        // Yksikködata välitetään kolmena NativeArray-taulukkona.
        [ReadOnly] public NativeArray<int> unitXCords;
        [ReadOnly] public NativeArray<int> unitYCords;
        [ReadOnly] public NativeArray<float> unitInfs;

        // Laskee ja palauttaa vaikutusarvon yhdelle kartan ruudulle.
        float CalculatePointInfluence(int col, int row)
        {
            float totalInfluence = 0.0f; // Kokonaisvaikutus.

            // Käydään läpi kaikki yksiköt ja summataan niiden vaikutus.
            for (int i = 0; i < unitInfs.Length; i++)
            {
                int x = unitXCords[i];
                int y = unitYCords[i];

                // Lasketaan yksikön etäisyys ruudusta.
                float dist = Distance(col, row, x, y);

                // Yksikön vaikutus ruutuun laskee etäisyyden mukaan.
                totalInfluence += unitInfs[i] / (1 + dist);
            }
            return totalInfluence;
        }

        public void Execute()
        {
            for (int x = 0; x < COLS; x++)
            {
                for (int y = 0; y < ROWS; y++)
                {
                    int index = x * ROWS + y;
                    // Lasketaan ruudun vaikutus ja tallennetaan se vaikutuskarttaan käsiteltävään indeksiin.
                    float influencePoint = CalculatePointInfluence(x, y);
                    influenceMap[index] = influencePoint;
                }
            }
        }
    }

    [SerializeField] GameObject influenceMapObj;
    [SerializeField] Renderer renderer;
    Texture2D influenceMapTexture;

    const int ROWS = 320;
    const int COLS = 320;

    const int minUnitsPerSide = 10;
    const int maxUnitsPerSide = 20;

    const float minInfluence = 1.0f;
    const float maxInfluence = 5.0f;

    int[,] offsets = new int[,] { { 0, 1 }, { 1, 1 }, { 1, 0 }, { 1, -1 }, { 0, -1 }, { -1, -1 }, { -1, 0 }, { -1, 1 } };
    float entityMoveSpeed = 0.1f;

    List<Unit> units;

    int unitCount = 0;
    [SerializeField] TMP_Text unitCountText;

    public float[] influenceMap;
    public float[] influenceMapPrev;


    List<int> unitXCords;
    List<int> unitYCords;
    List<float> unitInfs;


    public NativeArray<Unit> unitsNative;

    public NativeArray<int> unitXCordsNative;
    public NativeArray<int> unitYCordsNative;
    public NativeArray<float> unitInfsNative;


    public NativeArray<float> influenceMapNative;
    public NativeArray<float> influenceMapNativePrev;


    Color32 redColor = new Color32(255, 0, 0, 255);
    Color32 greenColor = new Color32(0, 255, 0, 255);

    int textureID;
    int textureColorID;

    JobHandle influenceMapJobHandle;
    InfluenceMapJob influenceMapJob;

    JobHandle influenceMapJobSTHandle;
    InfluenceMapJobST influenceMapJobST;

    JobHandle influenceMapJobVectorizedHandle;
    InfluenceMapJobVectorized influenceMapJobVectorized;

    JobHandle influenceMapJobUnitStructHandle;
    InfluenceMapJobUnitStruct influenceMapJobUnitStruct;

    bool influenceJobReady = false;

    Timer influenceMapTimer;

    [SerializeField] TMP_Text timerText;
    [SerializeField] TMP_Text renderTimerText;
    [SerializeField] TMP_Text mapSizeText;
    [SerializeField] TMP_Text entityCountText;

    public bool useNativeMap = false;

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
        units = new List<Unit>();

        unitXCords = new List<int>();
        unitYCords = new List<int>();
        unitInfs = new List<float>();

        influenceMap = new float[COLS * ROWS];
        influenceMapPrev = new float[COLS * ROWS];

        unitXCordsNative = new NativeArray<int>(1, Allocator.Persistent);
        unitYCordsNative = new NativeArray<int>(1, Allocator.Persistent);
        unitInfsNative = new NativeArray<float>(1, Allocator.Persistent);

        influenceMapNative = new NativeArray<float>(ROWS* COLS, Allocator.Persistent);
        influenceMapNativePrev = new NativeArray<float>(ROWS * COLS, Allocator.Persistent);

        influenceMapTexture = new Texture2D(ROWS, COLS, TextureFormat.RGBA32, false);
        influenceMapTexture.filterMode = FilterMode.Point;
        textureID = Shader.PropertyToID("_MainTex");
        textureColorID = Shader.PropertyToID("_Color");
        renderer.material.SetTexture(textureID, influenceMapTexture);
        renderer.material.SetColor(textureColorID, Color.white);

        
        mapSizeText.text = "Map size: " + COLS + " x " + ROWS;
        influenceMapTimer = new Timer();

        offsets = new int[,] { { 0, 1 }, { 1, 1 }, { 1, 0 }, { 1, -1 }, { 0, -1 }, { -1, -1 }, { -1, 0 }, { -1, 1 } };
    }

    
    public void InfluenceMapTest()
    {
        GenerateRandomUnitData(true);

        Debug.Log("Starting a new influence calc single threaded");

        influenceMapTimer.Restart();
        CalculateInfluenceMap();
        influenceMapTimer.Stop();

        timerText.text = "Influence map time: " + influenceMapTimer.GetTimeStr();

        influenceMap.CopyTo(influenceMapPrev, 0);
        useNativeMap = false;

        influenceMapTimer.Restart();
        RenderMapToTexture(influenceMap);
        influenceMapTimer.Stop();

        renderTimerText.text = "Render time: " + influenceMapTimer.GetTimeStr();
        //RenderNativeMapToTexture(influenceMapNativePrev, influenceMapTexture);
    }

    public void InfluenceMapStructTest()
    {
        GenerateRandomUnitsStruct();

        Debug.Log("Starting a new influence calc struct single threaded");

        Timer timer = new Timer();
        CalculateInfluenceMapStruct();
        timer.Stop();

        timerText.text = "Influence map time: " + timer.GetTimeStr();

        influenceMap.CopyTo(influenceMapPrev, 0);
        useNativeMap = false;

        RenderMapToTexture(influenceMap);
        //RenderNativeMapToTexture(influenceMapNativePrev, influenceMapTexture);
    }

    public void InfluenceMapJobTest()
    {
        // Generoidaan satunnainen yksikködata.
        GenerateRandomUnitDataNative(false);

        Debug.Log("Starting a new influenceMapJob");

        influenceMapTimer.Restart();

        // Jobin luonti.
        influenceMapJob = new InfluenceMapJob
        {
            influenceMap = influenceMapNative,
            unitXCords = unitXCordsNative,
            unitYCords = unitYCordsNative,
            unitInfs = unitInfsNative
        };

        // Ajoitetaan Jobi suoritettavaksi.
        influenceMapJobHandle = influenceMapJob.Schedule(ROWS * COLS, ROWS);

        // Varmistetaan Jobin valmistuminen.
        influenceMapJobHandle.Complete();

        influenceMapTimer.Stop();

        timerText.text = "Influence map time: " + influenceMapTimer.GetTimeStr();

        // Merkitään minkä tyyppista dataa käytetään, EntityManageria varten.
        useNativeMap = true;

        // Kopioidaan juuri laskettu kartta toiseen taulukkoon,
        // jotta sitä voidaan käyttää mahdollisessa tilanteessa,
        // jossa uuden kartan laskeminen on kesken.
        influenceMapNative.CopyTo(influenceMapNativePrev);

        // Visualisoidaan kartta tekstuuriin.
        RenderNativeMapToTexture(influenceMapNativePrev);
    }

    public void InfluenceMapJobSTTest()
    {
        // Generoidaan satunnainen yksikködata.
        GenerateRandomUnitDataNative(false);

        Debug.Log("Starting a new influenceMapJob");
        influenceMapTimer.Restart();

        // Jobin luonti.
        influenceMapJobST = new InfluenceMapJobST
        {
            influenceMap = influenceMapNative,
            unitXCords = unitXCordsNative,
            unitYCords = unitYCordsNative,
            unitInfs = unitInfsNative
        };

        // Ajoitetaan Jobi suoritettavaksi.
        influenceMapJobSTHandle = influenceMapJobST.Schedule();

        // Varmistetaan Jobin valmistuminen.
        influenceMapJobSTHandle.Complete();

        influenceMapTimer.Stop();
        timerText.text = "Influence map time: " + influenceMapTimer.GetTimeStr();

        // Merkitään minkä tyyppista dataa käytetään, EntityManageria varten.
        useNativeMap = true;

        // Kopioidaan juuri laskettu kartta toiseen taulukkoon,
        // jotta sitä voidaan käyttää mahdollisessa tilanteessa,
        // jossa uuden kartan laskeminen on kesken.
        influenceMapNative.CopyTo(influenceMapNativePrev);

        // Visualisoidaan kartta tekstuuriin.
        RenderNativeMapToTexture(influenceMapNativePrev);
    }
    public void InfluenceMapJobVectorizedTest()
    {
        // Generoidaan satunnainen yksikködata.
        GenerateRandomUnitDataNative(false);
        Debug.Log("Starting a new influenceMapJobVectorized");

        influenceMapTimer.Restart();

        // Vektorisoidun Jobin luonti.
        influenceMapJobVectorized = new InfluenceMapJobVectorized
        {
            influenceMap = influenceMapNative,
            unitXCords = unitXCordsNative,
            unitYCords = unitYCordsNative,
            unitInfs = unitInfsNative
        };

        // Ajoitetaan Jobi suoritettavaksi.
        influenceMapJobVectorizedHandle = influenceMapJobVectorized.Schedule(ROWS * COLS, ROWS);

        // Varmistetaan Jobin valmistuminen.
        influenceMapJobVectorizedHandle.Complete();

        influenceMapTimer.Stop();
        timerText.text = "Influence map time: " + influenceMapTimer.GetTimeStr();

        // Merkitään minkä tyyppista dataa käytetään, EntityManageria varten.
        useNativeMap = true;
        influenceMapNative.CopyTo(influenceMapNativePrev);

        // Visualisoidaan kartta tekstuuriin.
        RenderNativeMapToTexture(influenceMapNativePrev);
    }

    public void InfluenceMapJobStructTest()
    {
        // Generoidaan satunnainen yksikködata.
        GenerateRandomUnitsStructNative(true);
        Debug.Log("Starting a new influenceMapJobUnitStruct");

        influenceMapTimer.Restart();

        // Yksikkö tietueita käyttävän Jobin luonti.
        influenceMapJobUnitStruct = new InfluenceMapJobUnitStruct
        {
            influenceMap = influenceMapNative,
            units = unitsNative
        };

        // Ajoitetaan Jobi suoritettavaksi.
        influenceMapJobUnitStructHandle = influenceMapJobUnitStruct.Schedule(ROWS * COLS, ROWS);

        // Varmistetaan Jobin valmistuminen.
        influenceMapJobUnitStructHandle.Complete();

        influenceMapTimer.Stop();
        timerText.text = "Influence map time: " + influenceMapTimer.GetTimeStr();

        // Merkitään minkä tyyppista dataa käytetään, EntityManageria varten.
        useNativeMap = true;

        // Kopioidaan juuri laskettu kartta toiseen taulukkoon,
        // jotta sitä voidaan käyttää mahdollisessa tilanteessa,
        // jossa uuden kartan laskeminen on kesken.
        influenceMapNative.CopyTo(influenceMapNativePrev);

        // Visualisoidaan kartta tekstuuriin.
        RenderNativeMapToTexture(influenceMapNativePrev);
    }

    // Update is called once per frame
    void Update()
    {

        //if (!computeThreadJoined && computeReady)
        //{
        //    computeThread.Join();
        //    computeThreadJoined = true;
        //    jobTimer.PrintTime("Influence time: ");
        //    timerText.text = "Time:" + jobTimer.GetTimeStr();
        //    RenderMapToTexture(influenceMapTexture);
        //    Debug.Log("Compute thread joined!");
        //}

        unitCountText.text = "Unit count: " + unitCount.ToString();


        if (Input.GetKeyDown(KeyCode.Alpha1))
        {
            InfluenceMapTest();
        }

        if (Input.GetKeyDown(KeyCode.Alpha2))
        {
            InfluenceMapStructTest();
        }

        if (Input.GetKeyDown(KeyCode.Alpha3))
        {
            InfluenceMapJobTest();
        }

        if(Input.GetKeyDown(KeyCode.Alpha4))
        {
            InfluenceMapJobVectorizedTest();
        }

        if (Input.GetKeyDown(KeyCode.Alpha5))
        {
            InfluenceMapJobStructTest();
        }

        if (Input.GetKeyDown(KeyCode.Alpha6))
        {
            InfluenceMapJobSTTest();
        }
    }

    private void OnDestroy()
    {
        influenceMapNative.Dispose();
        influenceMapNativePrev.Dispose();

        unitXCordsNative.Dispose();
        unitYCordsNative.Dispose();
        unitInfsNative.Dispose();
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



    void GenerateRandomUnitData(bool deterministic = false)
    {
        // Satunnaislukugeneraattorille voidaan asettaa ennalta valittu siemenluku deterministisen kartan luomiseksi.
        if(deterministic)
        {
            Random.InitState(deterministicSeed);
        }

        // Valitaan luotavien yksiköiden määrä satunnaisesti.
        int unitsGreen = Random.Range(minUnitsPerSide, maxUnitsPerSide);
        int unitsRed = Random.Range(minUnitsPerSide, maxUnitsPerSide);

        int totalUnits = unitsGreen + unitsRed;

        unitXCords.Clear();
        unitYCords.Clear();
        unitInfs.Clear();

        // Luodaan yksiköt

        for (int i = 0; i < unitsGreen; i++)
        {
            int x = Random.Range(0, COLS - 1);
            int y = Random.Range(0, ROWS - 1);
            float influence = Random.Range(minInfluence, maxInfluence);

            unitXCords.Add(x);
            unitYCords.Add(y);
            unitInfs.Add(influence);
        }


        for (int i = unitsGreen; i < totalUnits; i++)
        {
            int x = Random.Range(0, COLS - 1);
            int y = Random.Range(0, ROWS - 1);
            // Punaisille yksiköille annetaan negatiivinen vaikutusarvo.
            float influence = Random.Range(minInfluence, maxInfluence) * -1.0f;

            unitXCords.Add(x);
            unitYCords.Add(y);
            unitInfs.Add(influence);
        }

        unitCount = totalUnits;
    }

    void GenerateRandomUnitDataNative(bool deterministic = false)
    {
        Debug.Log("Generating random native unit data...");

        // Satunnaislukugeneraattorille voidaan asettaa ennalta valittu siemenluku deterministisen kartan luomiseksi.
        if (deterministic)
        {
            Random.InitState(deterministicSeed);
        }

        // Valitaan luotavien yksiköiden määrä satunnaisesti.
        int unitsGreen = Random.Range(minUnitsPerSide, maxUnitsPerSide);
        int unitsRed = Random.Range(minUnitsPerSide, maxUnitsPerSide);
        
        int totalUnits = unitsGreen + unitsRed;

        // Vapautetaan aikaisemmin varatut taulukot muistivuotojen välttämiseksi.
        unitXCordsNative.Dispose();
        unitYCordsNative.Dispose();
        unitInfsNative.Dispose();

        // Varataan uudet NativeArrayt yksiköiden määrän mukaan.
        NativeArray<int> xCords = new NativeArray<int>(totalUnits, Allocator.Persistent);
        NativeArray<int> yCords = new NativeArray<int>(totalUnits, Allocator.Persistent);
        NativeArray<float> infs = new NativeArray<float>(totalUnits, Allocator.Persistent);

        // Asetetaan vapautetut osoittimet osoittamaan uusiin taulukkoihin.
        unitXCordsNative = xCords;
        unitYCordsNative = yCords;
        unitInfsNative = infs;

        // Luodaan yksiköt

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
            // Punaisille yksiköille annetaan negatiivinen vaikutusarvo.
            infs[i] = Random.Range(minInfluence, maxInfluence) * -1.0f;
        }

        unitCount = totalUnits;
        Debug.Log("Random units generated!");
    }

    void GenerateRandomUnitsStructNative(bool deterministic = false)
    {
        Debug.Log("Generating random units...");

        if (deterministic)
        {
            Random.InitState(deterministicSeed);
        }

        int unitsGreen = Random.Range(minUnitsPerSide, maxUnitsPerSide);
        int unitsRed = Random.Range(minUnitsPerSide, maxUnitsPerSide);

        int totalUnits = unitsGreen + unitsRed;

        NativeArray<Unit> newUnits = new NativeArray<Unit>(totalUnits, Allocator.Persistent);

        influenceMapJobUnitStructHandle.Complete();

        if(unitsNative.IsCreated)
        {
            unitsNative.Dispose();
        }
        
        unitsNative = newUnits;

        for (int i = 0; i < unitsGreen; i++)
        {
            Unit greenUnit = new Unit();
            int x = Random.Range(0, COLS - 1);
            int y = Random.Range(0, ROWS - 1);
            greenUnit.x = x;
            greenUnit.y = y;
            greenUnit.influence = Random.Range(minInfluence, maxInfluence);
            unitsNative[i] = greenUnit;
        }

        for (int i = unitsGreen; i < totalUnits; i++)
        {
            Unit redUnit = new Unit();
            int x = Random.Range(0, COLS - 1);
            int y = Random.Range(0, ROWS - 1);
            redUnit.x = x;
            redUnit.y = y;
            redUnit.influence = Random.Range(minInfluence, maxInfluence) * -1.0f;
            unitsNative[i] = redUnit;
        }

        unitCount = totalUnits;
        Debug.Log("Random units generated!");
    }

    void GenerateRandomUnitsStruct(bool deterministic = false)
    {
        Debug.Log("Generating random units struct...");

        if(deterministic)
        {
            Random.InitState(deterministicSeed);
        }

        int unitsGreen = Random.Range(minUnitsPerSide, maxUnitsPerSide);
        int unitsRed = Random.Range(minUnitsPerSide, maxUnitsPerSide);

        int totalUnits = unitsGreen + unitsRed;

        units.Clear();

        for (int i = 0; i < unitsGreen; i++)
        {
            Unit greenUnit = new Unit(); 
            int x = Random.Range(0, COLS - 1);
            int y = Random.Range(0, ROWS - 1);
            greenUnit.x = x;
            greenUnit.y = y;
            greenUnit.influence = Random.Range(minInfluence, maxInfluence);
            units.Add(greenUnit);
        }

        for (int i = unitsGreen; i < totalUnits; i++)
        {
            Unit redUnit = new Unit();
            int x = Random.Range(0, COLS - 1);
            int y = Random.Range(0, ROWS - 1);
            redUnit.x = x;
            redUnit.y = y;
            redUnit.influence = Random.Range(minInfluence, maxInfluence) * -1.0f;
            units.Add(redUnit);
        }

        unitCount = totalUnits;
        Debug.Log("Random units generated!");
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
        unitCount = 0;
        Debug.Log("Map reset!");
    }

    void ResetMap(float[] map)
    {
        Debug.Log("Resetting map...");
        for (int x = 0; x < COLS; x++)
        {
            for (int y = 0; y < COLS; y++)
            {
                int index = x * ROWS + y;
                map[index] = 0;
            }
        }
        unitCount = 0;
        Debug.Log("Map reset!");
    }

    // Calculates Euclidean distance between points
    static float Distance(int col0, int row0, int col1, int row1)
    {
        return Mathf.Sqrt((float)(col1 - col0) * (col1 - col0) + (row1 - row0) * (row1 - row0));
    }

    // Calculates Euclidean distance between points
    static float Distance(float col0, float row0, float col1, float row1)
    {
        return Mathf.Sqrt((float)(col1 - col0) * (col1 - col0) + (row1 - row0) * (row1 - row0));
    }

    // SOME INTRINSICS REQUIRE TO ENABLED IN PROJECT SETTINGS 
    static float horizontal_sumf(v256 x)
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

    static v256 CalcDistVector8f_v256(v256 c0, v256 r0, v256 c1, v256 r1)
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

    static float4 CalcDistFloat4(float4 c0, float4 r0, float4 c1, float4 r1)
    {
        return sqrt((c1 - c0) * (c1 - c0) + (r1 - r0) * (r1 - r0));
    }

    //static float4 CalcDistFloat4(float4 c0, float4 r0, float4 c1, float4 r1)
    //{
    //    return sqrt((c1 - c0) * (c1 - c0) + (r1 - r0) * (r1 - r0));

    //    //if (IsFmaSupported)
    //    //{
    //    //    float4 colSubVector = c1 - c0;

    //    //    float4 rowSubVector = r1 - r0;

    //    //    float4 rowMulVector = rowSubVector * rowSubVector;

    //    //    float4 sumVector = math.mad(colSubVector, colSubVector, rowMulVector);

    //    //    float4 sqrtVector = math.sqrt(sumVector);

    //    //    return sqrtVector;
    //    //}
    //    //else
    //    //{
    //    //    float4 colSubVector = c1 - c0;

    //    //    float4 rowSubVector = r1 - r0;

    //    //    float4 colMulVector = colSubVector * colSubVector;
    //    //    float4 rowMulVector = rowSubVector * rowSubVector;

    //    //    float4 sumVector = colMulVector + rowMulVector;


    //    //    float4 rsqrtVector = math.rsqrt(sumVector);
    //    //    float4 sqrtVector = math.rcp(rsqrtVector); // minimal loss of accuracy (max relative error: 1.5*2^-12)

    //    //    return sqrtVector;
    //    //}
    //}

    // Calculates and returns influence for a single point in the influence map
    float CalcPointInfStruct(int col, int row)
    {
        float totalInf = 0.0f;
        for (int i = 0; i < units.Count; i++)
        {
            int x = units[i].x;
            int y = units[i].y;

            float dist = Distance(col, row, x, y);

            totalInf += units[i].influence / (1 + dist);
        }
        return totalInf;
    }

    float CalculatePointInfluence(int col, int row)
    {
        float totalInfluence = 0.0f; // Kokonaisvaikutus.

        // Käydään läpi kaikki yksiköt ja summataan niiden vaikutus.
        for (int i = 0; i < unitInfs.Count; i++)
        {
            int x = unitXCords[i];
            int y = unitYCords[i];

            // Lasketaan yksikön etäisyys ruudusta.
            float distance = Distance(col, row, x, y);

            // Yksikön vaikutus ruutuun laskee etäisyyden mukaan.
            totalInfluence += unitInfs[i] / (1 + distance);
        }
        return totalInfluence;
    }

    void CalculateInfluenceMapStruct()
    {
        //Debug.Log("Calculating influence map...");
        //Timer influenceTimer = new Timer();

        for (int x = 0; x < COLS; x++)
        {
            for (int y = 0; y < ROWS; y++)
            {
                int index = x * ROWS + y;
                influenceMap[index] = CalcPointInfStruct(x, y);
            }
        }

        //influenceTimer.Stop();
        //influenceTimer.PrintTime("InfluenceTimer");
        //Debug.Log("Influence map calculated");
    }

    void CalculateInfluenceMap()
    {
        for (int x = 0; x < COLS; x++)
        {
            for (int y = 0; y < ROWS; y++)
            {
                int index = x * ROWS + y;
                influenceMap[index] = CalculatePointInfluence(x, y);
            }
        }
    }

    void RenderMapToTexture(float[] map)
    {
        // Käydään kaikki kartan ruudut läpi.
        for (int x = 0; x < COLS; x++)
        {
            for (int y = 0; y < ROWS; y++)
            {
                int index = x * ROWS + y;

                // Muutetaan värin läpinäkyvyyttä vaikutuksen itseisarvon mukaan.
                float influence = map[index]; 
                float alpha = 255 * Mathf.Abs(influence);

                // Puristetaan läpinäkyvyys välille 0-255.
                if (alpha < 0.0f)
                {
                    alpha = 0.0f;
                }
                else if (alpha > 255.0f)
                {
                    alpha = 255.0f;
                }

                // Asetetaan tekstuuriin vaikutusta vastaava pikseli.
                if (influence >= 0)
                {
                    greenColor.a = (byte)alpha;
                    influenceMapTexture.SetPixel(x, y, greenColor);
                }
                else
                {
                    redColor.a = (byte)alpha;
                    influenceMapTexture.SetPixel(x, y, redColor);
                }
            }
        }

        // Päivitetään tekstuuri.
        influenceMapTexture.Apply();
    }

    void RenderNativeMapToTexture(NativeArray<float> map)
    {
        Timer renderTimer = new Timer();
        for (int x = 0; x < COLS; x++)
        {
            for (int y = 0; y < ROWS; y++)
            {
                int index = x * ROWS + y;

                float influence = map[index];
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
                    influenceMapTexture.SetPixel(x, y, greenColor);
                }
                else
                {
                    redColor.a = (byte)alpha;
                    influenceMapTexture.SetPixel(x, y, redColor);
                }
            }
        }


        influenceMapTexture.Apply();
        renderer.material.SetTexture(textureID, influenceMapTexture);
        renderer.material.SetColor(textureColorID, Color.white);
        //renderTimer.PrintTime("RenderTimer");
        //Debug.Log("Rendered map to texture");
    }
}


