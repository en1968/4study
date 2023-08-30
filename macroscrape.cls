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

    ' �o�͐�̃V�[�g���쐬����
    Set outputSheet = ThisWorkbook.Sheets.Add(After:=ThisWorkbook.Sheets(ThisWorkbook.Sheets.Count))
    outputSheet.Name = "Extracted Values"

    ' �����Ԗڂ̃V�[�g���琔�l�𔲂��o��
    outputRow = 1 ' �o�͐�̍s��������
    outputCol = 1
    For Each startSheet In ThisWorkbook.Sheets
        If startSheet.Index Mod 2 = 0 Then ' �V�[�g�̃C���f�b�N�X���������ǂ������m�F
            outputCol = 1
            ' �����o�����l���擾
            cellValue = startSheet.Range("H8").Value
            outputSheet.Cells(outputRow, outputCol).Value = cellValue
            outputCol = outputCol + 1 ' �o�͐�̍s�����Ɉړ�

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