from tools.tools_llm import search_medic_info
from deep_translator import GoogleTranslator



# print(search_medic_info("What are the parameters for assessing nutritional status and growth in children with CKD?"))
text = """
The parameters for assessing nutritional status and growth in children with Chronic Kidney Disease (CKD), or what is referred to as PKG in the given text, include:

Assessing dietary intake: It's important to ensure that the child receives adequate nutrition through proper evaluation of their dietary intake. This assessment should be done every three days consecutively.
Height-for-age percentage according to age: Children with CKD have their height measured regularly (every month) to track normal growth patterns.
Growth velocity: Growth velocity refers to how quickly a child grows over time. In children with CKD, this needs regular monitoring.
Body weight and body mass index (BMI): Regular check on the child’s body weight and BMI helps identify potential imbalances, particularly water imbalance which can affect overall health in kids with CKD.
Head circumference: Periodic measurement of head circumference also forms part of the assessment parameters for these children.
The recommended restriction of sodium consumption is suggested for children with CKD experiencing early hypertension or high blood pressure.
Phosphate restrictions are beneficial for managing and preventing hyperparathyroidism, ensuring safe growth, nutrition, and bone mineralization.
Calcium recommendation ranges between 100% to 200% of daily requirements based on the child's age.
Vitamin D deficient patients are advised to receive Vitamin D supplementation with continuous monitoring of serum Vitamin D3 levels.
"""

print(GoogleTranslator(source='en', target='id').translate(text))