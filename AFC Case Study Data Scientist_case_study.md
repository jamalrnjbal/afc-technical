# Case Study: Data Scientist AI & ML – AFC Data & AI Team

## Kunden-Dashboard & Schadenprognose für Firmenwagenflotten

---

## Über AFC – Auto Fleet Control

**Auto Fleet Control (AFC)** bietet Schadenmanagementdienste für Firmenwagenflotten an und bearbeitet etwa **60.000 Schäden pro Jahr**. Dies ist Teil eines vollständig vertikal integrierten Dienstleistungsangebots, das vom individuellen Versicherungsschutz über das Schadenmanagement bis hin zur Reparatur (Karosserie und Glas) reicht.

**Rolle von Data & AI:** AFC setzt zunehmend auf KI-gestützte Technologien. Dazu gehören u.a. die automatisierte Schadenkalkulation über Bildanalyse, die intelligente Analyse aller Datenquellen (Kunden-, Fuhrpark-, Risiko- und Schadenmanagementdaten) sowie die Identifikation von Mustern in historischen Schadendaten. Das Team **Data & AI** entwickelt die datengetriebenen Lösungen, die diese Transformation vorantreiben.

---

## Kontext & Herausforderung

AFC plant ein neues **Dashboard** primär für Unternehmenskunden, um einen schnellen aggregierten Überblick über alte und offene Schadensmeldungen zu erhalten. Das Dashboard zeigt den Kostenverlauf und erwartete prognostizierte Schäden im verbleibenden Jahr. Das Sales-Team nutzt dasselbe Dashboard sekundär als Verkaufstool zur Demonstration des AFC-Mehrwerts bei der aktiven Schadenssteuerung.

Als Data Scientist im Team Data & AI arbeitest du an den analytischen und prädiktiven Komponenten, die dieses Dashboard mit Daten und Modellen versorgen.

---

## Deine Aufgabe

Dir steht ein Datensatz zur Verfügung:

| Datei | Inhalt | Zeilen |
|---|---|---|
| `claims_data.csv` | Schadensfälle mit Kosten, Kategorie, Kundenreferenz, Status | 2.000 |

### Datenbeschreibung: `claims_data.csv`

| Spalte | Typ | Beschreibung |
|---|---|---|
| `claim_id` | String | Eindeutige Schadensnummer |
| `customer_id` | String | Referenz zum Unternehmenskunden |
| `claim_date` | Datetime | Datum der Schadensmeldung |
| `vehicle_type` | String | Fahrzeugtyp (PKW, LKW, Transporter, Motorrad) |
| `damage_category` | String | Schadenskategorie (Karosserie, Motor, Elektronik, Glas, Fahrwerk, Interieur) |
| `fault_type` | String | Verschuldensart (Eigenverschulden, Fremdverschulden, Teilschuld, Vandalismus, Naturereignis) |
| `description` | String | Freitext-Schadensbeschreibung (teils leer) |
| `estimated_cost_eur` | Float | Geschätzte Kosten (€), teils fehlend |
| `actual_cost_eur` | Float | Tatsächliche Kosten (€) |
| `repair_duration_days` | Int | Reparaturdauer in Tagen, teils fehlend |
| `status` | String | Status (abgeschlossen, in_bearbeitung, wartend_auf_teile, storniert) |


---

## Aufgabenstellungen

Bearbeite die folgenden zwei Aufgaben. Du hast dafür **ca. 60 Minuten** Zeit. Wähle selbst, wie du deine Zeit aufteilst – es ist besser, alle Aufgaben anzugehen, als eine einzelne perfekt zu lösen.

---

### Aufgabe 1: Explorative Datenanalyse & Dashboard-Kennzahlen (ca. 20 Min.)

**Ziel**: Verstehe die Daten und entwickle die analytische Grundlage für das Kunden-Dashboard.

- Führe eine explorative Analyse durch: Wie verteilen sich Schäden über Zeit, Kunden, Kategorien und Verschuldensarten?
- Wie gehst du mit fehlenden Werten um? Begründe dein Vorgehen.
- Definiere und berechne **3–5 KPIs**, die im Dashboard für Unternehmenskunden sinnvoll wären.
- Überlege: Welche dieser KPIs eignen sich besonders, wenn das Sales-Team das Dashboard als Verkaufstool nutzt? Warum?

**Erwartung**: Code (Python), Visualisierungen, kurze Interpretation.

---

### Aufgabe 2: Schadenprognose für das verbleibende Jahr (ca. 25 Min.)

**Ziel**: Entwickle ein Modell, das die erwartete Schadenanzahl und/oder -kosten für einen Kunden im verbleibenden Jahr prognostiziert.

- Wähle einen geeigneten Ansatz: Zeitreihenprognose, Regressionsmodell oder einen anderen Ansatz deiner Wahl.
- Das Modell soll auf Kundenebene aggregiert arbeiten: „Wie viele Schäden und welche Kosten erwarten wir für Kunde X im Rest des Jahres 2026?"
- Bewerte dein Modell mit geeigneten Metriken.
- Beschreibe: Welche zusätzlichen Datenquellen würden die Prognose verbessern? Wie würdest du das Modell in Produktion bringen?

**Erwartung**: Code (Python), Modellbewertung, Diskussion der Annahmen und Limitationen.

---


## Abgabe

- **Format**: Jupyter Notebook (.ipynb) oder Python-Skript(e) + kurze Zusammenfassung
- **Präsentation**: Bereite eine **ca. 15-minütige Präsentation** deiner Ergebnisse vor, die anschließend gemeinsam diskutiert wird. Folien sind optional. Du kannst für deine Präsentation auch in einem Jupyter Notebook Code & deine Ergebnisse kombinierien. Erkläre deine Vorgehensweise, Ergebnisse und Empfehlungen so, dass auch nicht-technische Stakeholder sie verstehen.
- **Bewertungsfokus**: Wir achten nicht auf Perfektion, sondern auf:
  - Sauberen, lesbaren Code
  - Nachvollziehbare Entscheidungen und dokumentierte Annahmen
  - Fähigkeit, technische Ergebnisse in Business-Kontext zu übersetzen
  - Pragmatischen Umgang mit Datenqualitätsproblemen

---

## Hinweise

- Du darfst alle Tools und Libraries verwenden, die du möchtest.
- Mache geeignete Annahmen wo nötig – es geht um deinen Lösungsweg und deine Vorgehensweise.
- Wenn du Annahmen triffst, dokumentiere sie.
- Es gibt keine „perfekte" Lösung – uns interessiert dein Denkprozess.
- Fragen zur Aufgabenstellung kannst du jederzeit per E-Mail stellen.

---

*Viel Erfolg! Wir freuen uns auf deine Ergebnisse – und vielleicht schon bald auf dich als Teil unseres Teams in Hamburg.*
