using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Jobs;
using Unity.Burst;
using Unity.Collections;

public class ExampleScript : MonoBehaviour
{
    const int ROWS = 320;
    const int COLS = 320;


    [BurstCompile]
    struct InfluenceMapJob : IJobParallelFor
    {
        // Vaikutuskartta tallennetaan NativeArray-taulukkoon.
        public NativeArray<float> influenceMap;

        // Yksikk�data v�litet��n kolmena NativeArray-taulukkona.
        [ReadOnly] public NativeArray<int> unitXCords;
        [ReadOnly] public NativeArray<int> unitYCords;
        [ReadOnly] public NativeArray<float> unitInfs;

        // Laskee ja palauttaa vaikutusarvon yhdelle kartan ruudulle.
        float CalculatePointInfluence(int col, int row)
        {
            float totalInfluence = 0.0f; // Kokonaisvaikutus.

            // K�yd��n l�pi kaikki yksik�t ja summataan niiden vaikutus.
            for (int i = 0; i < unitInfs.Length; i++)
            {
                int x = unitXCords[i];
                int y = unitYCords[i];

                // Lasketaan yksik�n et�isyys ruudusta.
                float dist = Distance(col, row, x, y);

                // Yksik�n vaikutus ruutuun laskee et�isyyden mukaan.
                totalInfluence += unitInfs[i] / (1 + dist);
            }
            return totalInfluence;
        }

        // Kutsutaan yht� vaikutuskartan ruutua kohden.
        public void Execute(int index)
        {
            // Muunnetaan yksiulotteinen taulukon indeksi x- ja y-koordinaateiksi.
            int x = index / ROWS;
            int y = index % ROWS;

            // Lasketaan ruudun vaikutus ja tallennetaan se vaikutuskarttaan k�sitelt�v��n indeksiin.
            float influencePoint = CalculatePointInfluence(x, y);
            influenceMap[index] = influencePoint;
        }
    }

    // Start is called before the first frame update
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        
    }

    // Calculates Euclidean distance between points
    static float Distance(int col0, int row0, int col1, int row1)
    {
        return Mathf.Sqrt((float)(col1 - col0) * (col1 - col0) + (row1 - row0) * (row1 - row0));
    }


}
