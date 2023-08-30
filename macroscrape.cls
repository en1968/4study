VERSION 1.0 CLASS
BEGIN
  MultiUse = -1  'True
END
Attribute VB_Name = "Sheet1"
Attribute VB_GlobalNameSpace = False
Attribute VB_Creatable = False
Attribute VB_PredeclaredId = True
Attribute VB_Exposed = True
Sub ExtractValuesFromEvenSheets()
    Dim startSheet As Worksheet
    Dim outputSheet As Worksheet
    Dim cellValue As Variant
    Dim rowNum As Long
    Dim colNum As Long
    Dim outputRow As Long

    ' 出力先のシートを作成する
    Set outputSheet = ThisWorkbook.Sheets.Add(After:=ThisWorkbook.Sheets(ThisWorkbook.Sheets.Count))
    outputSheet.Name = "Extracted Values"

    ' 偶数番目のシートから数値を抜き出す
    outputRow = 1 ' 出力先の行を初期化
    outputCol = 1
    For Each startSheet In ThisWorkbook.Sheets
        If startSheet.Index Mod 2 = 0 Then ' シートのインデックスが偶数かどうかを確認
            outputCol = 1
            ' 抜き出す数値を取得
            cellValue = startSheet.Range("H8").Value
            outputSheet.Cells(outputRow, outputCol).Value = cellValue
            outputCol = outputCol + 1 ' 出力先の行を次に移動

            cellValue = startSheet.Range("H10").Value
            outputSheet.Cells(outputRow, outputCol).Value = cellValue
            outputCol = outputCol + 1

            cellValue = startSheet.Range("H12").Value
            outputSheet.Cells(outputRow, outputCol).Value = cellValue
            outputCol = outputCol + 1

            cellValue = startSheet.Range("H14").Value
            outputSheet.Cells(outputRow, outputCol).Value = cellValue
            outputCol = outputCol + 1

            cellValue = startSheet.Range("H16").Value
            outputSheet.Cells(outputRow, outputCol).Value = cellValue
            outputRow = outputRow + 1
        End If
    Next startSheet
End Sub
