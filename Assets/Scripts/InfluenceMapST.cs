using System.Collections;
using System.Collections.Generic;
using System.Threading;
using UnityEngine;
using Diagnostics = System.Diagnostics;

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

public class InfluenceMapST : MonoBehaviour
{
    [SerializeField] Renderer renderer;
    Texture2D influenceMapTexture;

    const int ROWS = 150;
    const int COLS = 150;

    const int minUnitsPerSide = 5;
    const int maxUnitsPerSide = 10;

    const float minInfluence = 1.0f;
    const float maxInfluence = 5.0f;


    float[,] unitsMap = new float[ROWS, COLS];
    float[,] influenceMap = new float[ROWS, COLS];

    Color32 redColor = new Color32(255, 0, 0, 255);
    Color32 greenColor = new Color32(0, 255, 0, 255);

    int textureID;
    int textureColorID;

    Thread computeThread = null;
    bool computeThreadJoined = true;
    bool computeReady = true;

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

            map[y, x] = Random.Range(minInfluence, maxInfluence);
        }


        for (int i = 0; i < unitsGreen; i++)
        {
            int x = Random.Range(0, COLS - 1);
            int y = Random.Range(0, ROWS - 1);

            map[y, x] = Random.Range(minInfluence, maxInfluence) * -1.0f;
        }

        Debug.Log("Random map generated!");
    }

    void ResetMap(float[,] map)
    {
        Debug.Log("Resetting map...");
        for (int i = 0; i < ROWS; i++)
        {
            for (int j = 0; j < COLS; j++)
            {
                map[i, j] = 0;
            }
        }
        Debug.Log("Map reset!");
    }

    float Distance(int row0, int col0, int row1, int col1)
    {
        return Mathf.Sqrt((float)(col0 - col1) * (col0 - col1) + (row0 - row1) * (row0 - row1));
    }

    float CalculateSquareInfluence(int x, int y)
    {
        float totalInfluence = 0.0f;
        for (int i = 0; i < ROWS; i++)
        {
            for (int j = 0; j < COLS; j++)
            { 
                float distance = Distance(y, x, j, i);
                //Debug.Log("Distance: " + distance);
                totalInfluence += unitsMap[i, j] / (1 + distance);
            }
        }

        return totalInfluence;
    }

    void CalculateInfluenceMap()
    {
        Debug.Log(Thread.CurrentThread.Name + " computing...");
        Debug.Log("Calculating influence map...");
        Timer influenceTimer = new Timer();
        for (int i = 0; i < ROWS; i++)
        {
            for (int j = 0; j < COLS; j++)
            {
                influenceMap[i, j] = CalculateSquareInfluence(j, i);
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
        for (int i = 0; i < ROWS; i++)
        {
            for (int j = 0; j < COLS; j++)
            {
                float alpha = 255 * Mathf.Abs(influenceMap[i, j]);

                if (alpha < 0.0f)
                {
                    alpha = 0.0f;
                }
                else if (alpha > 255.0f)
                {
                    alpha = 255.0f;
                }


                if (influenceMap[i, j] > 0)
                {
                    greenColor.a = (byte)alpha;
                    mapTexture.SetPixel(i, j, greenColor);
                }
                else
                {
                    redColor.a = (byte)alpha;
                    mapTexture.SetPixel(i, j, redColor);
                }
            }
        }


        mapTexture.Apply();
        renderer.material.SetTexture(textureID, mapTexture);
        renderer.material.SetColor(textureColorID, Color.white);
        //renderTimer.PrintTime("RenderTimer");
        //Debug.Log("Rendered map to texture");
    }




    // Start is called before the first frame update
    void Start()
    {
        influenceMapTexture = new Texture2D(ROWS, COLS, TextureFormat.RGBA32, false);
        influenceMapTexture.filterMode = FilterMode.Point;
        textureID = Shader.PropertyToID("_MainTex");
        textureColorID = Shader.PropertyToID("_Color");

        GenerateRandomMap(unitsMap);
        computeThread = new Thread(CalculateInfluenceMap);
        computeThread.Name = "ComputeThread";
        computeThreadJoined = false;
        computeReady = false;
        computeThread.Start();
    }

    public void StartComputeThread()
    {
        GenerateRandomMap(unitsMap);
        computeThread = new Thread(CalculateInfluenceMap);
        computeThread.Name = "ComputeThread";
        computeThreadJoined = false;
        computeReady = false;
        computeThread.Start();
    }

    // Update is called once per frame
    void Update()
    {
        if (!computeThreadJoined && computeReady)
        {
            computeThread.Join();
            computeThreadJoined = true;
            Debug.Log("Compute thread joined!");
        }

        if (computeThreadJoined && Input.GetKeyDown(KeyCode.Return))
        {
            StartComputeThread();
        }

        RenderMapToTexture(influenceMapTexture);
    }
}


