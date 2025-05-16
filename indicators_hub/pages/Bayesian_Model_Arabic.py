# test_dict_syntax.py

LANGUAGES = {
    "app_title": {"en": "Proposal: Adaptive Bayesian Estimation for Pilgrim Satisfaction Surveys",
                  "ar": "مقترح: تقدير Bayesian التكيفي لاستطلاعات رضا الحجاج"},
    "sidebar_title": {"en": "Proposal Sections", "ar": "أقسام المقترح"},
    "sidebar_lang_label": {"en": "Select Language", "ar": "اختر اللغة"},
    "sidebar_info": {"en": "This app presents a proposal for using Bayesian adaptive estimation for Hajj pilgrim satisfaction surveys. Developed by Dr. Mohammad Nabhan.",
                     "ar": "يقدم هذا التطبيق مقترحًا لاستخدام تقدير Bayesian التكيفي لاستطلاعات رضا الحجاج. تم التطوير بواسطة د. محمد نبهان."},

    # Section 1
    "sec1_title": {"en": "1. Introduction & Objectives", "ar": "١. مقدمة وأهداف"},
    "sec1_intro_md": {"en": """This proposal outlines an **Adaptive Bayesian Estimation framework** designed to enhance the process of gathering and analyzing survey data for Hajj pilgrim satisfaction and the assessment of services provided by various companies.

The current practice of developing satisfaction metrics month over month faces complexities, such as delays in pilgrim arrivals or non-uniformity across different months, making it difficult to consistently achieve high-confidence and low-error confidence intervals for key indicators. This proposal aims to introduce a more dynamic, efficient, and robust methodology.
    """, "ar": """يحدد هذا المقترح إطارًا لـ **تقدير Bayesian التكيفي** مصممًا لتعزيز عملية جمع وتحليل بيانات الاستطلاع الخاصة برضا حجاج بيت الله الحرام وتقييم الخدمات المقدمة من مختلف الشركات.

تواجه الممارسة الحالية المتمثلة في تطوير مقاييس الرضا شهرًا بعد شهر تعقيدات، مثل التأخير في وصول الحجاج أو عدم الانتظام عبر الأشهر المختلفة، مما يجعل من الصعب تحقيق فترات ثقة عالية الدقة ومنخفضة الخطأ للمؤشرات الرئيسية بشكل مستمر. يهدف هذا المقترح إلى تقديم منهجية أكثر ديناميكية وكفاءة وقوة."""},
    "sec1_obj_subheader": {"en": "1.1. Primary Objectives", "ar": "١.١. الأهداف الأساسية"},
    "sec1_obj_md": {"en": """
The core objectives of implementing this adaptive Bayesian framework are:

* **Achieve Desired Precision Efficiently:** To obtain satisfaction metrics and service provider assessments with pre-defined levels of precision (i.e., narrow credible intervals at a specific confidence level) using optimized sample sizes.
* **Dynamic Sampling Adjustments:** To iteratively adjust sampling efforts based on accumulating evidence. This means collecting more data only when and where it's needed to meet precision targets, avoiding over-sampling or under-sampling.
* **Timely and Reliable Estimates:** To provide decision-makers with more timely and statistically robust estimates, allowing for quicker responses to emerging issues or trends in pilgrim satisfaction.
* **Incorporate Prior Knowledge:** To formally integrate knowledge from previous survey waves, historical data, or expert opinions into the estimation process, leading to more informed starting points and potentially faster convergence to precise estimates.
* **Adapt to Changing Conditions:** To develop a system that can adapt to changes in satisfaction levels or service provider performance over time, for instance, by adjusting the influence of older data.
* **Enhanced Subgroup Analysis:** To facilitate more reliable analysis of specific pilgrim subgroups or service aspects by adaptively ensuring sufficient data is collected for these segments.
    """, "ar": """
الأهداف الأساسية لتطبيق إطار Bayesian التكيفي هي:

* **تحقيق الدقة المطلوبة بكفاءة:** الحصول على مقاييس الرضا وتقييمات مقدمي الخدمات بمستويات دقة محددة مسبقًا (أي فترات موثوقية ضيقة عند مستوى ثقة معين) باستخدام أحجام عينات مُحسَّنة.
* **تعديلات أخذ العينات الديناميكية:** تعديل جهود أخذ العينات بشكل متكرر بناءً على الأدلة المتراكمة. هذا يعني جمع المزيد من البيانات فقط عند الحاجة إليها وحيثما تكون هناك حاجة إليها لتحقيق أهداف الدقة، وتجنب الإفراط في أخذ العينات أو نقصها.
* **تقديرات موثوقة وفي الوقت المناسب:** تزويد صانعي القرار بتقديرات أكثر موثوقية من الناحية الإحصائية وفي الوقت المناسب، مما يسمح باستجابات أسرع للقضايا الناشئة أو الاتجاهات في رضا الحجاج.
* **دمج المعرفة المسبقة:** دمج المعرفة من موجات الاستطلاع السابقة أو البيانات التاريخية أو آراء الخبراء رسميًا في عملية التقدير، مما يؤدي إلى نقاط انطلاق أكثر استنارة واحتمال تقارب أسرع للتقديرات الدقيقة.
* **التكيف مع الظروف المتغيرة:** تطوير نظام يمكنه التكيف مع التغيرات في مستويات الرضا أو أداء مقدمي الخدمات بمرور الوقت، على سبيل المثال، عن طريق تعديل تأثير البيانات القديمة.
* **تحليل محسن للمجموعات الفرعية:** تسهيل تحليل أكثر موثوقية لمجموعات فرعية محددة من الحجاج أو جوانب الخدمة من خلال ضمان جمع بيانات كافية لهذه الشرائح بشكل تكيفي."""},

    # Section 2
    "sec2_title": {"en": "2. Challenges Addressed by this Methodology", "ar": "٢. التحديات التي تعالجها هذه المنهجية"},
    "sec2_intro_md": {"en": "The proposed Bayesian adaptive estimation framework directly addresses several key challenges currently faced in the Hajj survey process:",
                      "ar": "يعالج إطار تقدير Bayesian التكيفي المقترح بشكل مباشر العديد من التحديات الرئيسية التي تواجه حاليًا في عملية استطلاع الحج:"},
    "sec2_challenge1_header": {"en": "Difficulty in Obtaining Stable Confidence Intervals", "ar": "صعوبة الحصول على فترات ثقة مستقرة"},
    "sec2_challenge1_md": {"en": """
* **Challenge:** Operational complexities like staggered pilgrim arrivals, varying visa availability periods, and diverse pilgrim schedules lead to non-uniform data collection across time. This makes it hard to achieve consistent and narrow confidence intervals for satisfaction indicators using fixed sampling plans.
* **Bayesian Solution:** The adaptive nature allows sampling to continue until a desired precision (credible interval width) is met, regardless of initial data flow irregularities. Estimates stabilize as more data is incorporated.
    """, "ar": """
* **التحدي:** التعقيدات التشغيلية مثل وصول الحجاج المتدرج، وفترات توفر التأشيرات المتفاوتة، وجداول الحجاج المتنوعة تؤدي إلى جمع بيانات غير منتظم عبر الزمن. هذا يجعل من الصعب تحقيق فترات ثقة متسقة وضيقة لمؤشرات الرضا باستخدام خطط أخذ عينات ثابتة.
* **حل Bayesian:** تسمح الطبيعة التكيفية باستمرار أخذ العينات حتى يتم تحقيق الدقة المطلوبة (عرض فترة الموثوقية)، بغض النظر عن عدم انتظام تدفق البيانات الأولي. تستقر التقديرات مع دمج المزيد من البيانات."""},
    "sec2_challenge2_header": {"en": "Inefficiency of Fixed Sample Size Approaches", "ar": "عدم كفاءة مناهج حجم العينة الثابت"},
    "sec2_challenge2_md": {"en": """
* **Challenge:** Predetermined sample sizes often lead to either over-sampling (wasting resources when satisfaction is homogenous or already precisely estimated) or under-sampling (resulting in inconclusive results or wide confidence intervals).
* **Bayesian Solution:** Sampling effort is guided by the current level of uncertainty. If an estimate is already precise, sampling can be reduced or stopped for that segment. If it's imprecise, targeted additional sampling is guided by the model.
    """, "ar": """
* **التحدي:** غالبًا ما تؤدي أحجام العينات المحددة مسبقًا إما إلى الإفراط في أخذ العينات (إهدار الموارد عندما يكون الرضا متجانسًا أو تم تقديره بدقة بالفعل) أو نقص أخذ العينات (مما يؤدي إلى نتائج غير حاسمة أو فترات ثقة واسعة).
* **حل Bayesian:** يتم توجيه جهد أخذ العينات حسب المستوى الحالي من عدم اليقين. إذا كان التقدير دقيقًا بالفعل، فيمكن تقليل أخذ العينات أو إيقافه لتلك الشريحة. إذا كان غير دقيق، يتم توجيه أخذ عينات إضافية مستهدفة بواسطة النموذج."""},
    "sec2_challenge3_header": {"en": "Incorporation of Prior Knowledge and Historical Data", "ar": "دمج المعرفة المسبقة والبيانات التاريخية"},
    "sec2_challenge3_md": {"en": """
* **Challenge:** Valuable insights from past surveys or existing knowledge about certain pilgrim groups or services are often not formally used to inform current survey efforts or baseline estimates.
* **Bayesian Solution:** Priors provide a natural mechanism to incorporate such information. This can lead to more accurate estimates, especially when current data is sparse, and can make the learning process more efficient.
    """, "ar": """
* **التحدي:** غالبًا لا يتم استخدام الرؤى القيمة من الاستطلاعات السابقة أو المعرفة الحالية حول مجموعات معينة من الحجاج أو الخدمات رسميًا لإبلاغ جهود الاستطلاع الحالية أو التقديرات الأساسية.
* **حل Bayesian:** توفر التوزيعات المسبقة (Priors) آلية طبيعية لدمج هذه المعلومات. يمكن أن يؤدي ذلك إلى تقديرات أكثر دقة، خاصة عندما تكون البيانات الحالية متفرقة، ويمكن أن يجعل عملية التعلم أكثر كفاءة."""},
    "sec2_challenge4_header": {"en": "Assessing Service Provider Performance with Evolving Data", "ar": "تقييم أداء مقدمي الخدمات مع تطور البيانات"},
    "sec2_challenge4_md": {"en": """
* **Challenge:** Evaluating service providers is difficult when their performance might change over time, or when initial data for a new provider is limited. Deciding when enough data has been collected to make a fair assessment is crucial.
* **Bayesian Solution:** The framework can be designed to track performance iteratively. For new providers, it starts with less informative priors and builds evidence. For existing ones, it can incorporate past performance, potentially with mechanisms to down-weight older data if performance is expected to evolve (see Section 3.5).
    """, "ar": """
* **التحدي:** يصعب تقييم مقدمي الخدمات عندما قد يتغير أداؤهم بمرور الوقت، أو عندما تكون البيانات الأولية لمقدم خدمة جديد محدودة. يعد تحديد متى تم جمع بيانات كافية لإجراء تقييم عادل أمرًا بالغ الأهمية.
* **حل Bayesian:** يمكن تصميم الإطار لتتبع الأداء بشكل متكرر. بالنسبة لمقدمي الخدمات الجدد، يبدأ بتوزيعات مسبقة أقل إفادة ويبني الأدلة. بالنسبة للموجودين، يمكنه دمج الأداء السابق، مع آليات محتملة لتقليل وزن البيانات القديمة إذا كان من المتوقع أن يتطور الأداء (انظر القسم ٣.٥)."""},
    "sec2_challenge5_header": {"en": "Why Not Just Combine All Data (Data Lumping)?", "ar": "لماذا لا نكتفي بدمج جميع البيانات (تجميع البيانات)؟"},
    "sec2_challenge5_md": {"en": """
A common question is: "How is this different from simply lumping old data with newly acquired data and calculating a new confidence interval based on the combined dataset?"

While seemingly straightforward, this 'data lumping' approach has significant drawbacks compared to the structured Bayesian adaptive framework:

* **Ignores Changes Over Time (Non-Stationarity):** Pilgrim satisfaction and service provider performance are not static. Older data may no longer accurately reflect current conditions. Simply combining data treats all observations as equally relevant, which can be misleading if underlying trends have shifted. A major logistics issue last year, if lumped in, could falsely depress current satisfaction metrics even if resolved.
* **Lack of Formal Weighting for Data Relevance:** Data lumping gives equal weight to every data point, regardless of its age or the context in which it was collected. The Bayesian framework, through its prior-posterior updating and mechanisms like discount factors (see Section 3.5), allows for a principled way to give more influence to recent data or to systematically reduce the influence of older, potentially outdated information.
* **Misleading Precision:** Combining heterogeneous data can lead to confidence/credible intervals that appear narrow (high precision) but are actually averaging over different underlying states. This can mask real changes or current problems, giving a false sense of stability or accuracy.
* **No Mechanism for Gradual Learning or Adaptation:** The Bayesian approach is inherently iterative. It learns from new data batches and updates its 'beliefs' (posterior distributions) systematically. This allows for a more nuanced understanding of how estimates are evolving. Data lumping is a one-off recalculation that doesn't offer this dynamic learning perspective.
* **Difficulty in Incorporating Expert Knowledge Formally:** The Bayesian framework explicitly uses prior distributions to incorporate expert opinion or findings from related studies *before* seeing new data. Data lumping lacks a formal mechanism for this initial grounding.

**In essence, the Bayesian adaptive approach provides a statistically sound and flexible method to:**
1.  Formally incorporate prior knowledge.
2.  Sequentially update estimates as new data arrives.
3.  Systematically manage the influence of older data to adapt to changing conditions.
4.  Guide sampling efficiently towards achieving specific precision targets.

Simple data lumping misses these crucial adaptive and structural advantages, potentially leading to less reliable or even misleading conclusions in dynamic environments.
    """, "ar": """
سؤال شائع هو: "كيف يختلف هذا عن مجرد تجميع البيانات القديمة مع البيانات المكتسبة حديثًا وحساب فترة ثقة جديدة بناءً
على مجموعة البيانات المجمعة؟"

على الرغم من أن نهج "تجميع البيانات" هذا يبدو مباشرًا، إلا أنه يحتوي على عيوب كبيرة مقارنة بإطار Bayesian التكيفي المنظم:

* **يتجاهل التغييرات بمرور الوقت (عدم الاستقرار):** رضا الحجاج وأداء مقدمي الخدمات ليسا ثابتين. قد لا تعكس البيانات القديمة الظروف الحالية بدقة. إن مجرد دمج البيانات يعامل جميع الملاحظات على أنها ذات صلة متساوية، وهو ما يمكن أن يكون مضللاً إذا تغيرت الاتجاهات الأساسية. مشكلة لوجستية كبيرة العام الماضي، إذا تم تجميعها، يمكن أن تخفض بشكل خاطئ مقاييس الرضا الحالية حتى لو تم حلها.
* **الافتقار إلى آلية ترجيح رسمية لأهمية البيانات:** يعطي تجميع البيانات وزنًا متساويًا لكل نقطة بيانات، بغض النظر عن عمرها أو السياق الذي تم جمعها فيه. يسمح إطار Bayesian، من خلال تحديثه المسبق اللاحق وآليات مثل عوامل الخصم (انظر القسم ٣.٥)، بطريقة مبدئية لإعطاء تأثير أكبر للبيانات الحديثة أو لتقليل تأثير المعلومات القديمة التي قد تكون قديمة بشكل منهجي.
* **دقة مضللة:** يمكن أن يؤدي دمج البيانات غير المتجانسة إلى فترات ثقة/موثوقية تبدو ضيقة (دقة عالية) ولكنها في الواقع متوسطة عبر حالات أساسية مختلفة. هذا يمكن أن يخفي التغييرات الحقيقية أو المشاكل الحالية، مما يعطي إحساسًا خاطئًا بالاستقرار أو الدقة.
* **لا توجد آلية للتعلم التدريجي أو التكيف:** نهج Bayesian تكراري بطبيعته. يتعلم من دفعات البيانات الجديدة ويحدث "معتقداته" (التوزيعات اللاحقة) بشكل منهجي. هذا يسمح بفهم أكثر دقة لكيفية تطور التقديرات. تجميع البيانات هو إعادة حساب لمرة واحدة لا تقدم هذا المنظور التعليمي الديناميكي.
* **صعوبة دمج معرفة الخبراء رسميًا:** يستخدم إطار Bayesian صراحة التوزيعات المسبقة لدمج رأي الخبراء أو النتائج من الدراسات ذات الصلة *قبل* رؤية البيانات الجديدة. يفتقر تجميع البيانات إلى آلية رسمية لهذا الأساس الأولي.

**في جوهره، يوفر نهج Bayesian التكيفي طريقة سليمة إحصائيًا ومرنة من أجل:**
١. دمج المعرفة المسبقة رسميًا.
٢. تحديث التقديرات بالتتابع عند وصول بيانات جديدة.
٣. إدارة تأثير البيانات القديمة بشكل منهجي للتكيف مع الظروف المتغيرة.
٤. توجيه أخذ العينات بكفاءة نحو تحقيق أهداف دقة محددة.

يفتقد تجميع البيانات البسيط هذه المزايا التكيفية والهيكلية الحاسمة، مما قد يؤدي إلى استنتاجات أقل موثوقية أو حتى مضللة في البيئات الديناميكية."""},

    # Section 3
    "sec3_title": {"en": "3. Core Methodology: Bayesian Adaptive Estimation", "ar": "٣. المنهجية الأساسية: تقدير Bayesian التكيفي"},
    "sec3_intro_md": {"en": "The Bayesian adaptive estimation framework is an iterative process that leverages Bayes' theorem to update our beliefs about pilgrim satisfaction or service performance as new survey data is collected. This allows for dynamic adjustments to the sampling strategy.",
                       "ar": "إطار تقدير Bayesian التكيفي هو عملية تكرارية تستفيد من نظرية Bayes لتحديث معتقداتنا حول رضا الحجاج أو أداء الخدمة عند جمع بيانات استطلاع جديدة. هذا يسمح بإجراء تعديلات ديناميكية على استراتيجية أخذ العينات."},
    "sec3_concepts_subheader": {"en": "3.1. Fundamental Concepts", "ar": "٣.١. المفاهيم الأساسية"},
    "sec3_concepts_md": {"en": r"""
At its heart, Bayesian inference combines prior knowledge with observed data to arrive at an updated understanding, known as the posterior distribution.

* **Prior Distribution ($P(\theta)$):** This represents our initial belief about a parameter $\theta$ (e.g., the proportion of satisfied pilgrims) *before* observing new data. It can be based on historical data, expert opinion, or be deliberately "uninformative" if we want the data to speak for itself.
* **Likelihood ($P(D|\theta)$):** This quantifies how probable the observed data ($D$) is, given a particular value of the parameter $\theta$. It is the function that connects the data to the parameter.
* **Posterior Distribution ($P(\theta|D)$):** This is our updated belief about $\theta$ *after* observing the data. It is calculated using Bayes' Theorem:
    $$ P(\theta|D) = \frac{P(D|\theta) \times P(\theta)}{P(D)} $$
    Where $P(D)$ is the marginal likelihood of the data, acting as a normalizing constant. In practice, we often focus on the proportionality:
    $$ P(\theta|D) \propto P(D|\theta) \times P(\theta) $$
* **Credible Interval:** In Bayesian statistics, a credible interval is a range of values that contains the parameter $\theta$ with a certain probability (e.g., 95%). This is a direct probabilistic statement about the parameter, unlike the frequentist confidence interval.
    """, "ar": r"""
في جوهره، يجمع استدلال Bayesian بين المعرفة المسبقة والبيانات المرصودة للوصول إلى فهم محدث، يُعرف بالتوزيع اللاحق.

* **التوزيع المسبق ($P(\theta)$):** يمثل هذا اعتقادنا الأولي حول معلمة $\theta$ (على سبيل المثال، نسبة الحجاج الراضين) *قبل* ملاحظة البيانات الجديدة. يمكن أن يستند إلى البيانات التاريخية أو رأي الخبراء أو أن يكون "غير إعلامي" بشكل متعمد إذا أردنا أن تتحدث البيانات عن نفسها.
* **دالة الإمكان ($P(D|\theta)$):** تحدد هذه الدالة مدى احتمالية البيانات المرصودة ($D$)، بالنظر إلى قيمة معينة للمعلومة $\theta$. إنها الدالة التي تربط البيانات بالمعلمة.
* **التوزيع اللاحق ($P(\theta|D)$):** هذا هو اعتقادنا المحدث حول $\theta$ *بعد* ملاحظة البيانات. يتم حسابه باستخدام نظرية Bayes:
    $$ P(\theta|D) = \frac{P(D|\theta) \times P(\theta)}{P(D)} $$
    حيث $P(D)$ هو الإمكان الهامشي للبيانات، ويعمل كثابت تسوية. في الممارسة العملية، غالبًا ما نركز على التناسب:
    $$ P(\theta|D) \propto P(D|\theta) \times P(\theta) $$
* **فترة الموثوقية:** في إحصاءات Bayesian، فترة الموثوقية هي نطاق من القيم يحتوي على المعلمة $\theta$ باحتمال معين (على سبيل المثال، ٩٥٪). هذا بيان احتمالي مباشر حول المعلمة، على عكس فترة الثقة الإحصائية التقليدية."""},
    "sec3_iterative_subheader": {"en": "3.2. The Iterative Process", "ar": "٣.٢. العملية التكرارية"},
    "sec3_iterative_md": {"en": """
The adaptive methodology follows these steps:
1.  **Initialization:**
    * Define the parameter(s) of interest (e.g., satisfaction with lodging, food, logistics for a specific company).
    * Specify an initial **prior distribution** for each parameter. For satisfaction proportions, a Beta distribution is commonly used.
    * Set a target precision (e.g., a maximum width for the 95% credible interval).

2.  **Initial Data Collection:**
    * Collect an initial batch of survey responses relevant to the parameter(s). The size of this initial batch can be based on practical considerations or a small fixed number.

3.  **Posterior Update:**
    * Use the collected data (likelihood) and the current prior distribution to calculate the **posterior distribution** for each parameter.

4.  **Precision Assessment:**
    * Calculate the credible interval from the posterior distribution.
    * Compare the width of this interval to the target precision.

5.  **Adaptive Decision & Iteration:**
    * **If Target Precision Met:** For the given parameter, the current level of precision is sufficient. Sampling for this specific indicator/segment can be paused or stopped. The current posterior distribution provides the estimate and its uncertainty.
    * **If Target Precision Not Met:** More data is needed.
        * Determine an appropriate additional sample size. This can be guided by projecting how the credible interval width might decrease with more data (based on the current posterior).
        * Collect the additional batch of survey responses.
        * Return to Step 3 (Posterior Update), using the current posterior as the new prior for the next update.

This cycle continues until the desired precision is achieved for all key indicators or available resources for the current wave are exhausted.
    """, "ar": """
تتبع المنهجية التكيفية هذه الخطوات:
١. **التهيئة:**
    * تحديد المعلمة (المعلمات) ذات الأهمية (على سبيل المثال، الرضا عن السكن، الطعام، الخدمات اللوجستية لشركة معينة).
    * تحديد **توزيع مسبق** أولي لكل معلمة. بالنسبة لنسب الرضا، يشيع استخدام توزيع Beta.
    * تحديد دقة مستهدفة (على سبيل المثال، أقصى عرض لفترة موثوقية بنسبة ٩٥٪).

٢. **جمع البيانات الأولي:**
    * جمع دفعة أولية من ردود الاستطلاع ذات الصلة بالمعلمة (المعلمات). يمكن أن يعتمد حجم هذه الدفعة الأولية على اعتبارات عملية أو عدد ثابت صغير.

٣. **تحديث التوزيع اللاحق:**
    * استخدام البيانات المجمعة (دالة الإمكان) والتوزيع المسبق الحالي لحساب **التوزيع اللاحق** لكل معلمة.

٤. **تقييم الدقة:**
    * حساب فترة الموثوقية من التوزيع اللاحق.
    * مقارنة عرض هذه الفترة بالدقة المستهدفة.

٥. **القرار التكيفي والتكرار:**
    * **إذا تم تحقيق الدقة المستهدفة:** بالنسبة للمعلمة المحددة، يكون المستوى الحالي من الدقة كافيًا. يمكن إيقاف أخذ العينات مؤقتًا أو إيقافه لهذا المؤشر/الشريحة المحددة. يوفر التوزيع اللاحق الحالي التقدير وعدم اليقين فيه.
    * **إذا لم يتم تحقيق الدقة المستهدفة:** هناك حاجة إلى مزيد من البيانات.
        * تحديد حجم عينة إضافي مناسب. يمكن توجيه ذلك من خلال توقع كيف قد ينخفض عرض فترة الموثوقية مع المزيد من البيانات (بناءً على التوزيع اللاحق الحالي).
        * جمع الدفعة الإضافية من ردود الاستطلاع.
        * العودة إلى الخطوة ٣ (تحديث التوزيع اللاحق)، باستخدام التوزيع اللاحق الحالي كتوزيع مسبق جديد للتحديث التالي.

تستمر هذه الدورة حتى يتم تحقيق الدقة المطلوبة لجميع المؤشرات الرئيسية أو استنفاد الموارد المتاحة للموجة الحالية."""},
    "sec3_modeling_subheader": {"en": "3.3. Modeling Satisfaction (e.g., using Beta-Binomial Model)",
                                "ar": "٣.٣. نمذجة الرضا (على سبيل المثال، باستخدام نموذج Beta-Binomial)"},
    "sec3_modeling_md": {"en": r"""
For satisfaction metrics that are proportions (e.g., percentage of pilgrims rating a service as "satisfied" or "highly satisfied"), the Beta-Binomial model is highly suitable and commonly used.

* **Parameter of Interest ($\theta$):** The true underlying proportion of satisfied pilgrims.
* **Prior Distribution (Beta):** We assume the prior belief about $\theta$ follows a Beta distribution, denoted as $Beta(\alpha_0, \beta_0)$.
    * $\alpha_0 > 0$ and $\beta_0 > 0$ are the parameters of the prior.
    * An uninformative prior could be $Beta(1, 1)$, which is equivalent to a Uniform(0,1) distribution.
    * Prior knowledge can be incorporated by setting $\alpha_0$ and $\beta_0$ based on historical data (e.g., $\alpha_0$ = past successes, $\beta_0$ = past failures).
* **Likelihood (Binomial/Bernoulli):** If we collect $n$ new responses, and $k$ of them are "satisfied" (successes), the likelihood of observing $k$ successes in $n$ trials is given by the Binomial distribution:
    $$ P(k, n | \theta) = \binom{n}{k} \theta^k (1-\theta)^{n-k} $$
* **Posterior Distribution (Beta):** Due to the conjugacy between the Beta prior and Binomial likelihood, the posterior distribution of $\theta$ is also a Beta distribution:
    $$ \theta | k, n \sim Beta(\alpha_{post}, \beta_{post}) = Beta(\alpha_0 + k, \beta_0 + n - k) $$
    So, the updated parameters are $\alpha_{post} = \alpha_0 + k$ and $\beta_{post} = \beta_0 + n - k$.

* **Point Estimate (Posterior Mean):** The mean of this posterior distribution is often used as the point estimate for the satisfaction proportion $\theta$:
    $$ E[\theta | D] = \frac{\alpha_{post}}{\alpha_{post} + \beta_{post}} = \frac{\alpha_0 + k}{\alpha_0 + k + \beta_0 + n - k} $$
* **Credible Interval:** The $(1-\gamma)\%$ credible interval for $\theta$ is typically computed numerically from the posterior Beta distribution $Beta(\alpha_{post}, \beta_{post})$ using its cumulative distribution function (CDF) or percent point function (PPF). For example, a 95% credible interval provides a range within which $\theta$ lies with 95% probability, given the data and the prior. This is calculated in Python using `scipy.stats.beta.interval(0.95, alpha_posterior, beta_posterior)`.

This conjugacy simplifies calculations significantly.
    """, "ar": r"""
بالنسبة لمقاييس الرضا التي هي عبارة عن نسب (على سبيل المثال، النسبة المئوية للحجاج الذين يقيمون خدمة بأنها "مرضية" أو "مرضية للغاية")، فإن نموذج Beta-Binomial مناسب جدًا ويستخدم بشكل شائع.

* **المعلمة ذات الأهمية ($\theta$):** النسبة الأساسية الحقيقية للحجاج الراضين.
* **التوزيع المسبق (Beta):** نفترض أن الاعتقاد المسبق حول $\theta$ يتبع توزيع Beta، يُشار إليه بالرمز $Beta(\alpha_0, \beta_0)$.
    * $\alpha_0 > 0$ و $\beta_0 > 0$ هما معلمات التوزيع المسبق.
    * يمكن أن يكون التوزيع المسبق غير الإعلامي $Beta(1, 1)$، وهو ما يعادل توزيع Uniform(0,1).
    * يمكن دمج المعرفة المسبقة عن طريق تعيين $\alpha_0$ و $\beta_0$ بناءً على البيانات التاريخية (على سبيل المثال، $\alpha_0$ = النجاحات السابقة، $\beta_0$ = الإخفاقات السابقة).
* **دالة الإمكان (Binomial/Bernoulli):** إذا جمعنا $n$ ردودًا جديدة، وكان $k$ منها "راضين" (نجاحات)، فإن إمكانية ملاحظة $k$ نجاحات في $n$ محاولات تُعطى بواسطة توزيع Binomial:
    $$ P(k, n | \theta) = \binom{n}{k} \theta^k (1-\theta)^{n-k} $$
* **التوزيع اللاحق (Beta):** نظرًا للترافق بين توزيع Beta المسبق ودالة الإمكان Binomial، فإن التوزيع اللاحق لـ $\theta$ هو أيضًا توزيع Beta:
    $$ \theta | k, n \sim Beta(\alpha_{post}, \beta_{post}) = Beta(\alpha_0 + k, \beta_0 + n - k) $$
    لذا، فإن المعلمات المحدثة هي $\alpha_{post} = \alpha_0 + k$ و $\beta_{post} = \beta_0 + n - k$.

* **التقدير النقطي (المتوسط اللاحق):** غالبًا ما يستخدم متوسط هذا التوزيع اللاحق كتقدير نقطي لنسبة الرضا $\theta$:
    $$ E[\theta | D] = \frac{\alpha_{post}}{\alpha_{post} + \beta_{post}} = \frac{\alpha_0 + k}{\alpha_0 + k + \beta_0 + n - k} $$
* **فترة الموثوقية:** عادةً ما يتم حساب فترة الموثوقية $(1-\gamma)\%$ لـ $\theta$ عدديًا من توزيع Beta اللاحق $Beta(\alpha_{post}, \beta_{post})$ باستخدام دالة التوزيع التراكمي (CDF) أو دالة نقطة النسبة المئوية (PPF). على سبيل المثال، توفر فترة موثوقية بنسبة ٩٥٪ نطاقًا يقع ضمنه $\theta$ باحتمال ٩٥٪، بالنظر إلى البيانات والتوزيع المسبق. يتم حساب ذلك في Python باستخدام `scipy.stats.beta.interval(0.95, alpha_posterior, beta_posterior)`.

هذا الترافق يبسط الحسابات بشكل كبير."""},
    "sec3_sampling_logic_subheader": {"en": "3.4. Adaptive Sampling Logic & Determining Additional Sample Size",
                                      "ar": "٣.٤. منطق أخذ العينات التكيفي وتحديد حجم العينة الإضافي"},
    "sec3_sampling_logic_md": {"en": r"""
The decision to continue sampling is based on whether the current credible interval for $\theta$ meets the desired precision.

* **Stopping Rule:** Stop sampling for a specific metric when (for a $(1-\gamma)\%$ credible interval $[L, U]$):
    $$ U - L \leq \text{Target Width} $$
    And/or when the credible interval lies entirely above/below a certain threshold of practical importance.

* **Estimating Required Additional Sample Size (Conceptual):**
    While exact formulas for sample size to guarantee a future credible interval width are complex because the width itself is a random variable, several approaches can guide this:
    1.  **Simulation:** Based on the current posterior $Beta(\alpha_{post}, \beta_{post})$, simulate drawing additional samples of various sizes. For each simulated sample size, calculate the resulting posterior and its credible interval width. This can give a distribution of expected widths for different additional $n$.
    2.  **Approximation Formulas:** Some researchers have developed approximations. For instance, one common approach for proportions aims for a certain margin of error (half-width) $E_{target}$ in the credible interval. If the current variance of the posterior is $Var(\theta | D_{current})$, and we approximate the variance of the posterior after $n_{add}$ additional samples as roughly $\frac{Var(\theta | D_{current}) \times N_0}{N_0 + n_{add}}$ (where $N_0 = \alpha_{post} + \beta_{post}$ is the "effective prior sample size"), one can solve for $n_{add}$ that makes the future standard deviation (and thus interval width) small enough.
    3.  **Bayesian Sequential Analysis:** More formal methods from Bayesian sequential analysis (e.g., Bayesian sequential probability ratio tests - BSPRTs) can be adapted, though they might be more complex to implement initially.
    4.  **Pragmatic Batching:** Collect data in smaller, manageable batches (e.g., 30-50 responses). After each batch, reassess precision. This is often a practical starting point.

The tool should aim to provide guidance on a reasonable next batch size based on the current uncertainty and the distance to the target precision.
    """, "ar": r"""
يعتمد قرار مواصلة أخذ العينات على ما إذا كانت فترة الموثوقية الحالية لـ $\theta$ تفي بالدقة المطلوبة.

* **قاعدة الإيقاف:** إيقاف أخذ العينات لمقياس معين عندما (لفترة موثوقية $(1-\gamma)\%$ $[L, U]$):
    $$ U - L \leq \text{العرض المستهدف} $$
    و/أو عندما تقع فترة الموثوقية بالكامل فوق/تحت عتبة معينة ذات أهمية عملية.

* **تقدير حجم العينة الإضافي المطلوب (مفاهيمي):**
    في حين أن الصيغ الدقيقة لحجم العينة لضمان عرض فترة موثوقية مستقبلية معقدة لأن العرض نفسه متغير عشوائي، يمكن لعدة مناهج توجيه ذلك:
    ١. **المحاكاة:** بناءً على التوزيع اللاحق الحالي $Beta(\alpha_{post}, \beta_{post})$، قم بمحاكاة سحب عينات إضافية بأحجام مختلفة. لكل حجم عينة محاكاة، احسب التوزيع اللاحق الناتج وعرض فترة الموثوقية الخاصة به. يمكن أن يعطي هذا توزيعًا للعروض المتوقعة لـ $n$ إضافية مختلفة.
    ٢. **صيغ تقريبية:** طور بعض الباحثين تقريبات. على سبيل المثال، يهدف نهج شائع للنسب إلى هامش خطأ معين (نصف العرض) $E_{target}$ في فترة الموثوقية. إذا كان التباين الحالي للتوزيع اللاحق هو $Var(\theta | D_{current})$، وقمنا بتقريب تباين التوزيع اللاحق بعد $n_{add}$ عينات إضافية على أنه تقريبًا $\frac{Var(\theta | D_{current}) \times N_0}{N_0 + n_{add}}$ (حيث $N_0 = \alpha_{post} + \beta_{post}$ هو "حجم العينة المسبقة الفعال")، يمكن للمرء حل $n_{add}$ الذي يجعل الانحراف المعياري المستقبلي (وبالتالي عرض الفترة) صغيرًا بدرجة كافية.
    ٣. **تحليل Bayesian التسلسلي:** يمكن تكييف طرق أكثر رسمية من تحليل Bayesian التسلسلي (على سبيل المثال، اختبارات نسبة الاحتمالية التسلسلية لـ Bayesian - BSPRTs)، على الرغم من أنها قد تكون أكثر تعقيدًا في التنفيذ الأولي.
    ٤. **التجميع العملي:** جمع البيانات في دفعات أصغر يمكن التحكم فيها (على سبيل المثال، ٣٠-٥٠ ردًا). بعد كل دفعة، أعد تقييم الدقة. غالبًا ما يكون هذا نقطة انطلاق عملية.

يجب أن تهدف الأداة إلى تقديم إرشادات حول حجم الدفعة التالية المعقول بناءً على عدم اليقين الحالي والمسافة إلى الدقة المستهدفة."""},
    "sec3_heterogeneity_subheader": {"en": "3.5. Handling Data Heterogeneity Over Time",
                                     "ar": "٣.٥. التعامل مع عدم تجانس البيانات بمرور الوقت"},
    "sec3_heterogeneity_md": {"en": """
A key challenge is that service provider performance or general pilgrim satisfaction might change over time. Using historical data uncritically as a prior might be misleading if changes have occurred.

* **The "Learning Hyperparameter" (Discount Factor / Power Prior):**
    One way to address this is to down-weight older data. If we have a series of data batches $D_1, D_2, \dots, D_t$ (from oldest to newest), when forming a prior for the current period $t+1$ based on data up to $t$, we can use a "power prior" approach or a simpler discount factor.
    For example, if using the posterior from period $t$ (with parameters $\alpha_t, \beta_t$) as a prior for period $t+1$, we might introduce a discount factor $\delta \in [0, 1]$:
    $$ \alpha_{prior, t+1} = \delta \times \alpha_t + (1-\delta) \times \alpha_{initial} $$
    $$ \beta_{prior, t+1} = \delta \times \beta_t + (1-\delta) \times \beta_{initial} $$
    Where $(\alpha_{initial}, \beta_{initial})$ could be parameters of a generic, uninformative prior.
    * If $\delta = 1$, all past information is carried forward fully.
    * If $\delta = 0$, all past information is discarded, and we restart with the initial prior (re-estimation).
    * Values between 0 and 1 provide a trade-off. The "learning hyperparameter" $\delta$ can be fixed, tuned, or even learned from the data if a more complex model is used. A simpler approach is to use a fixed $\delta$, e.g., $\delta=0.8$ or $\delta=0.9$, reflecting a belief that recent data is more relevant.

* **Change-Point Detection:**
    Periodically, statistical tests can be run to detect if there has been a significant change in the underlying satisfaction or performance metric. If a change point is detected (e.g., using CUSUM charts on posterior means, or more formal Bayesian change-point models), the prior for subsequent estimations might be reset to be less informative, or data before the change point heavily discounted or discarded.

* **Hierarchical Bayesian Models (Advanced):**
    These models can explicitly model variation over time or across different service providers simultaneously, allowing "borrowing strength" across units while also estimating individual trajectories. This is a more sophisticated approach suitable for later phases.

The choice of method depends on the complexity deemed appropriate and the available data. Starting with a discount factor is often a pragmatic first step.
    """, "ar": """
التحدي الرئيسي هو أن أداء مقدمي الخدمات أو رضا الحجاج العام قد يتغير بمرور الوقت. قد يكون استخدام البيانات التاريخية بشكل غير نقدي كتوزيع مسبق مضللاً إذا حدثت تغييرات.

* **"المعلمة الفائقة للتعلم" (عامل الخصم / Power Prior):**
    إحدى طرق معالجة ذلك هي تقليل وزن البيانات القديمة. إذا كان لدينا سلسلة من دفعات البيانات $D_1, D_2, \dots, D_t$ (من الأقدم إلى الأحدث)، عند تكوين توزيع مسبق للفترة الحالية $t+1$ بناءً على البيانات حتى $t$， يمكننا استخدام نهج "power prior" أو عامل خصم أبسط.
    على سبيل المثال، إذا تم استخدام التوزيع اللاحق من الفترة $t$ (بالمعلمات $\alpha_t, \beta_t$) كتوزيع مسبق للفترة $t+1$， فقد نقدم عامل خصم $\delta \in [0, 1]$:
    $$ \alpha_{prior, t+1} = \delta \times \alpha_t + (1-\delta) \times \alpha_{initial} $$
    $$ \beta_{prior, t+1} = \delta \times \beta_t + (1-\delta) \times \beta_{initial} $$
    حيث $(\alpha_{initial}, \beta_{initial})$ يمكن أن تكون معلمات لتوزيع مسبق عام غير إعلامي.
    * إذا كانت $\delta = 1$， يتم نقل جميع المعلومات السابقة بالكامل.
    * إذا كانت $\delta = 0$， يتم تجاهل جميع المعلومات السابقة، ونبدأ من جديد بالتوزيع المسبق الأولي (إعادة التقدير).
    * توفر القيم بين 0 و 1 مقايضة. يمكن تثبيت "المعلمة الفائقة للتعلم" $\delta$ أو ضبطها أو حتى تعلمها من البيانات إذا تم استخدام نموذج أكثر تعقيدًا. النهج الأبسط هو استخدام $\delta$ ثابتة، على سبيل المثال، $\delta=0.8$ أو $\delta=0.9$， مما يعكس اعتقادًا بأن البيانات الحديثة أكثر صلة.

* **كشف نقطة التغيير:**
    بشكل دوري، يمكن إجراء اختبارات إحصائية للكشف عما إذا كان هناك تغيير كبير في مقياس الرضا أو الأداء الأساسي. إذا تم الكشف عن نقطة تغيير (على سبيل المثال، باستخدام مخططات CUSUM على المتوسطات اللاحقة، أو نماذج Bayesian أكثر رسمية لنقاط التغيير)، فقد يتم إعادة تعيين التوزيع المسبق للتقديرات اللاحقة ليكون أقل إفادة، أو يتم خصم البيانات قبل نقطة التغيير بشدة أو تجاهلها.

* **نماذج Bayesian الهرمية (متقدمة):**
    يمكن لهذه النماذج نمذجة التباين بمرور الوقت أو عبر مختلف مقدمي الخدمات في وقت واحد، مما يسمح "باقتراض القوة" عبر الوحدات مع تقدير المسارات الفردية أيضًا. هذا نهج أكثر تطوراً ومناسب للمراحل اللاحقة.

يعتمد اختيار الطريقة على التعقيد الذي يعتبر مناسبًا والبيانات المتاحة. غالبًا ما يكون البدء بعامل الخصم خطوة عملية أولى."""},

    # Section 4
    "sec4_title": {"en": "4. Implementation Roadmap (Conceptual)", "ar": "٤. خارطة طريق التنفيذ (مفاهيمية)"},
    "sec4_intro_md": {"en": "Implementing the Bayesian adaptive estimation framework involves several key stages:",
                       "ar": "يتضمن تنفيذ إطار تقدير Bayesian التكيفي عدة مراحل رئيسية:"},
    "roadmap_phase_col": {"en": "Phase", "ar": "المرحلة"},
    "roadmap_step_col": {"en": "Step", "ar": "الخطوة"},
    "roadmap_desc_col": {"en": "Description", "ar": "الوصف"},
    "roadmap_data": {
        "en": [
            ["Phase 1: Foundation & Pilot", "1. Define Key Metrics & Precision Targets", "Identify critical satisfaction indicators and service aspects. For each, define the desired level of precision (e.g., 95% credible interval width of ±3%)."],
            ["Phase 1: Foundation & Pilot", "2. System Setup & Prior Elicitation", "Establish data collection pathways. For each metric, determine initial priors (e.g., $Beta(1,1)$ for uninformative, or derive from historical averages if stable and relevant)."],
            ["Phase 2: Iterative Development & Testing", "3. Model Development & Initial Batching Logic", "Develop the Bayesian models (e.g., Beta-Binomial) for core metrics. Implement the logic for posterior updates and initial rules for determining subsequent sample batch sizes."],
            ["Phase 2: Iterative Development & Testing", "4. Dashboard Development & Pilot Testing", "Create a dashboard to visualize posterior distributions, credible intervals, precision achieved vs. target, and sampling progress. Conduct a pilot study on a limited scale to test the workflow, model performance, and adaptive logic."],
            ["Phase 3: Full-Scale Deployment & Refinement", "5. Scaled Rollout & Heterogeneity Modeling", "Gradually roll out the adaptive system across more survey areas/service providers. Implement or refine mechanisms for handling data heterogeneity over time (e.g., discount factors, change-point monitoring)."],
            ["Phase 3: Full-Scale Deployment & Refinement", "6. Continuous Monitoring & Improvement", "Continuously monitor the system's performance, resource efficiency, and the quality of estimates. Refine models, priors, and adaptive rules based on ongoing learning and feedback."]
        ],
        "ar": [
            ["المرحلة الأولى: التأسيس والتجربة الأولية", "١. تحديد المقاييس الرئيسية وأهداف الدقة", "تحديد مؤشرات الرضا الحاسمة والجوانب الخدمية. لكل منها، تحديد مستوى الدقة المطلوب (على سبيل المثال، عرض فترة موثوقية بنسبة ٩٥٪ يبلغ ±٣٪)."],
            ["المرحلة الأولى: التأسيس والتجربة الأولية", "٢. إعداد النظام واستنباط التوزيعات المسبقة", "إنشاء مسارات جمع البيانات. لكل مقياس، تحديد التوزيعات المسبقة الأولية (على سبيل المثال، $Beta(1,1)$ لغير الإعلامي، أو اشتقاقها من المتوسطات التاريخية إذا كانت مستقرة وذات صلة)."],
            ["المرحلة الثانية: التطوير التكراري والاختبار", "٣. تطوير النموذج ومنطق التجميع الأولي", "تطوير نماذج Bayesian (على سبيل المثال، Beta-Binomial) للمقاييس الأساسية. تنفيذ منطق التحديثات اللاحقة والقواعد الأولية لتحديد أحجام دفعات العينات اللاحقة."],
            ["المرحلة الثانية: التطوير التكراري والاختبار", "٤. تطوير لوحة المعلومات والاختبار التجريبي", "إنشاء لوحة معلومات لتصور التوزيعات اللاحقة، وفترات الموثوقية، والدقة المحققة مقابل الهدف، وتقدم أخذ العينات. إجراء دراسة تجريبية على نطاق محدود لاختبار سير العمل وأداء النموذج والمنطق التكيفي."],
            ["المرحلة الثالثة: النشر على نطاق واسع والتحسين", "٥. التوسع التدريجي ونمذجة عدم التجانس", "التوسع التدريجي للنظام التكيفي عبر المزيد من مناطق الاستطلاع/مقدمي الخدمات. تنفيذ أو تحسين آليات التعامل مع عدم تجانس البيانات بمرور الوقت (على سبيل المثال، عوامل الخصم، مراقبة نقاط التغيير)."],
            ["المرحلة الثالثة: النشر على نطاق واسع والتحسين", "٦. المراقبة المستمرة والتحسين", "مراقبة أداء النظام وكفاءة الموارد وجودة التقديرات بشكل مستمر. تحسين النماذج والتوزيعات المسبقة والقواعد التكيفية بناءً على التعلم المستمر والملاحظات."]
        ]
    },

    # Section 5
    "sec5_title": {"en": "5. Note to Practitioners", "ar": "٥. ملاحظة للممارسين"},
    "sec5_benefits_subheader": {"en": "5.1. Benefits of the Bayesian Adaptive Approach", "ar": "٥.١. فوائد نهج Bayesian التكيفي"},
    "sec5_benefits_md": {"en": """
* **Efficiency:** Targets sampling effort where it's most needed, potentially reducing overall sample sizes compared to fixed methods while achieving desired precision.
* **Adaptability:** Responds to incoming data, making it suitable for dynamic environments where satisfaction might fluctuate or where initial knowledge is low.
* **Formal Use of Prior Knowledge:** Allows systematic incorporation of historical data or expert insights, which can be particularly useful with sparse initial data for new services or specific subgroups.
* **Intuitive Uncertainty Quantification:** Credible intervals offer a direct probabilistic interpretation of the parameter's range, which can be easier for stakeholders to understand than frequentist confidence intervals.
* **Rich Output:** Provides a full posterior distribution for each parameter, offering more insight than just a point estimate and an interval.
    """, "ar": """
* **الكفاءة:** يستهدف جهد أخذ العينات حيث تشتد الحاجة إليه، مما قد يقلل من أحجام العينات الإجمالية مقارنة بالطرق الثابتة مع تحقيق الدقة المطلوبة.
* **القدرة على التكيف:** يستجيب للبيانات الواردة، مما يجعله مناسبًا للبيئات الديناميكية حيث قد يتقلب الرضا أو حيث تكون المعرفة الأولية منخفضة.
* **الاستخدام الرسمي للمعرفة المسبقة:** يسمح بالدمج المنهجي للبيانات التاريخية أو رؤى الخبراء، والتي يمكن أن تكون مفيدة بشكل خاص مع البيانات الأولية المتفرقة للخدمات الجديدة أو المجموعات الفرعية المحددة.
* **قياس عدم اليقين بشكل بديهي:** توفر فترات الموثوقية تفسيرًا احتماليًا مباشرًا لنطاق المعلمة، والذي يمكن أن يكون أسهل على أصحاب المصلحة فهمه من فترات الثقة الإحصائية التقليدية.
* **مخرجات غنية:** يوفر توزيعًا لاحقًا كاملاً لكل معلمة، مما يوفر رؤية أعمق من مجرد تقدير نقطي وفترة."""},
    "sec5_limitations_subheader": {"en": "5.2. Limitations and Considerations", "ar": "٥.٢. القيود والاعتبارات"},
    "sec5_limitations_md": {"en": """
* **Complexity:** Bayesian methods can be conceptually more demanding than traditional frequentist approaches. Implementation requires specialized knowledge.
* **Prior Selection:** The choice of prior distribution can influence posterior results, especially with small sample sizes. This requires careful justification and transparency. While "uninformative" priors aim to minimize this influence, truly uninformative priors are not always straightforward.
* **Computational Cost:** While Beta-Binomial models are computationally simple, more complex Bayesian models (e.g., hierarchical models, models requiring MCMC simulation) can be computationally intensive.
* **Interpretation Differences:** Practitioners familiar with frequentist statistics need to understand the different interpretations of Bayesian outputs (e.g., credible intervals vs. confidence intervals).
* **"Black Box" Perception:** If not explained clearly, the adaptive nature and Bayesian calculations might be perceived as a "black box" by those unfamiliar with the methods. Clear communication is key.
    """, "ar": """
* **التعقيد:** يمكن أن تكون طرق Bayesian أكثر تطلبًا من الناحية المفاهيمية من الأساليب الإحصائية التقليدية. يتطلب التنفيذ معرفة متخصصة.
* **اختيار التوزيع المسبق:** يمكن أن يؤثر اختيار التوزيع المسبق على النتائج اللاحقة، خاصة مع أحجام العينات الصغيرة. يتطلب هذا تبريرًا دقيقًا وشفافية. بينما تهدف التوزيعات المسبقة "غير الإعلامية" إلى تقليل هذا التأثير، فإن التوزيعات المسبقة غير الإعلامية حقًا ليست دائمًا واضحة ومباشرة.
* **التكلفة الحسابية:** بينما تكون نماذج Beta-Binomial بسيطة حسابيًا، يمكن أن تكون نماذج Bayesian الأكثر تعقيدًا (على سبيل المثال، النماذج الهرمية، النماذج التي تتطلب محاكاة MCMC) كثيفة حسابيًا.
* **اختلافات التفسير:** يحتاج الممارسون المطلعون على الإحصاءات التقليدية إلى فهم التفسيرات المختلفة لمخرجات Bayesian (على سبيل المثال، فترات الموثوقية مقابل فترات الثقة).
* **تصور "الصندوق الأسود":** إذا لم يتم شرحها بوضوح، فقد يُنظر إلى الطبيعة التكيفية وحسابات Bayesian على أنها "صندوق أسود" من قبل أولئك غير المطلعين على الطرق. التواصل الواضح هو المفتاح."""},
    "sec5_assumptions_subheader": {"en": "5.3. Key Assumptions", "ar": "٥.٣. الافتراضات الرئيسية"},
    "sec5_assumptions_md": {"en": """
* **Representativeness of Samples:** Each batch of collected data is assumed to be representative of the (sub)population of interest *at that point in time*. Sampling biases will affect the validity of estimates.
* **Model Appropriateness:** The chosen likelihood and prior distributions should reasonably reflect the data-generating process and existing knowledge. For satisfaction proportions, the Beta-Binomial model is often robust.
* **Stability (or Modeled Change):** The underlying parameter being measured (e.g., satisfaction rate) is assumed to be relatively stable between iterative updates within a survey wave, OR changes are explicitly modeled (e.g., via discount factors or dynamic models). Rapid, unmodeled fluctuations can be challenging.
* **Accurate Data:** Assumes responses are truthful and accurately recorded.
    """, "ar": """
* **تمثيلية العينات:** يُفترض أن كل دفعة من البيانات المجمعة تمثل المجموعة (الفرعية) السكانية ذات الأهمية *في تلك اللحظة الزمنية*. ستؤثر تحيزات أخذ العينات على صحة التقديرات.
* **ملاءمة النموذج:** يجب أن تعكس دالة الإمكان والتوزيعات المسبقة المختارة بشكل معقول عملية توليد البيانات والمعرفة الحالية. بالنسبة لنسب الرضا، غالبًا ما يكون نموذج Beta-Binomial قويًا.
* **الاستقرار (أو التغيير المنمذج):** يُفترض أن المعلمة الأساسية التي يتم قياسها (على سبيل المثال، معدل الرضا) مستقرة نسبيًا بين التحديثات التكرارية داخل موجة الاستطلاع، أو يتم نمذجة التغييرات بشكل صريح (على سبيل المثال، عبر عوامل الخصم أو النماذج الديناميكية). يمكن أن تكون التقلبات السريعة غير المنمذجة صعبة.
* **بيانات دقيقة:** يفترض أن الردود صادقة ومسجلة بدقة."""},
    "sec5_recommendations_subheader": {"en": "5.4. Practical Recommendations", "ar": "٥.٤. توصيات عملية"},
    "sec5_recommendations_md": {"en": """
* **Start Simple:** Begin with core satisfaction metrics and simple models (like Beta-Binomial). Complexity can be added iteratively as experience is gained.
* **Invest in Training:** Ensure that the team involved in implementing and interpreting the results has adequate training in Bayesian statistics.
* **Transparency is Key:** Document choices for priors, models, and adaptive rules. Perform sensitivity analyses to understand the impact of different prior choices, especially in early stages or with limited data.
* **Regular Review and Validation:** Periodically review the performance of the models. Compare Bayesian estimates with those from traditional methods if possible, especially during a transition period. Validate assumptions.
* **Stakeholder Communication:** Develop clear ways to communicate the methodology, its benefits, and the interpretation of results to stakeholders who may not be statisticians.
* **Pilot Thoroughly:** Before full-scale implementation, conduct thorough pilot studies to refine the process, test the technology, and identify unforeseen challenges.
    """, "ar": """
* **ابدأ ببساطة:** ابدأ بمقاييس الرضا الأساسية والنماذج البسيطة (مثل Beta-Binomial). يمكن إضافة التعقيد بشكل متكرر مع اكتساب الخبرة.
* **استثمر في التدريب:** تأكد من أن الفريق المشارك في تنفيذ وتفسير النتائج لديه تدريب كافٍ في إحصاءات Bayesian.
* **الشفافية هي المفتاح:** قم بتوثيق الخيارات الخاصة بالتوزيعات المسبقة والنماذج والقواعد التكيفية. قم بإجراء تحليلات الحساسية لفهم تأثير خيارات التوزيع المسبق المختلفة، خاصة في المراحل المبكرة أو مع بيانات محدودة.
* **المراجعة والتحقق المنتظم:** قم بمراجعة أداء النماذج بشكل دوري. قارن تقديرات Bayesian بالطرق التقليدية إذا أمكن، خاصة خلال فترة انتقالية. تحقق من صحة الافتراضات.
* **التواصل مع أصحاب المصلحة:** طور طرقًا واضحة لتوصيل المنهجية وفوائدها وتفسير النتائج لأصحاب المصلحة الذين قد لا يكونون إحصائيين.
* **التجربة الأولية الشاملة:** قبل التنفيذ على نطاق واسع، قم بإجراء دراسات تجريبية شاملة لتحسين العملية واختبار التكنولوجيا وتحديد التحديات غير المتوقعة."""},

    # Section 6
    "sec6_title": {"en": "6. Interactive Illustration: Beta-Binomial Model", "ar": "٦. توضيح تفاعلي: نموذج Beta-Binomial"},
    "sec6_intro_md": {"en": "This section provides a simple interactive illustration of how a Beta prior is updated to a Beta posterior with new data (Binomial likelihood). This is the core of estimating a proportion (e.g., satisfaction rate) in a Bayesian way.",
                      "ar": "يقدم هذا القسم توضيحًا تفاعليًا بسيطًا لكيفية تحديث توزيع Beta المسبق إلى توزيع Beta لاحق ببيانات جديدة (دالة الإمكان Binomial). هذا هو جوهر تقدير نسبة (على سبيل المثال، معدل الرضا) بطريقة Bayesian."},
    "sec6_prior_subheader": {"en": "Prior Beliefs", "ar": "المعتقدات المسبقة"},
    "sec6_prior_md": {"en": "The Beta distribution $Beta(\\alpha, \\beta)$ is a common prior for proportions. $\\alpha$ can be thought of as prior 'successes' and $\\beta$ as prior 'failures'. $Beta(1,1)$ is a uniform (uninformative) prior.",
                       "ar": "توزيع Beta $Beta(\\alpha, \\beta)$ هو توزيع مسبق شائع للنسب. يمكن اعتبار $\\alpha$ بمثابة "نجاحات" مسبقة و $\\beta$ بمثابة "إخفاقات" مسبقة. $Beta(1,1)$ هو توزيع منتظم (غير إعلامي) مسبق."},
    "sec6_prior_alpha_label": {"en": "Prior Alpha (α₀)", "ar": "ألفا المسبقة (α₀)"},
    "sec6_prior_beta_label": {"en": "Prior Beta (β₀)", "ar": "بيتا المسبقة (β₀)"},
    "sec6_prior_mean_label": {"en": "Prior Mean", "ar": "المتوسط المسبق"},
    "sec6_prior_ci_label": {"en": "95% Credible Interval (Prior)", "ar": "فترة الموثوقية ٩٥٪ (المسبقة)"},
    "sec6_width_label": {"en": "Width", "ar": "العرض"},
    "sec6_likelihood_subheader": {"en": "New Survey Data (Likelihood)", "ar": "بيانات الاستطلاع الجديدة (دالة الإمكان)"},
    "sec6_likelihood_md": {"en": "Enter the results from a new batch of surveys.", "ar": "أدخل النتائج من دفعة جديدة من الاستطلاعات."},
    "sec6_surveys_n_label": {"en": "Number of New Surveys (n)", "ar": "عدد الاستطلاعات الجديدة (n)"},
    "sec6_surveys_k_label": {"en": "Number Satisfied in New Surveys (k)", "ar": "عدد الراضين في الاستطلاعات الجديدة (k)"},
    "sec6_observed_sat_label": {"en": "Observed Satisfaction in New Data", "ar": "الرضا الملاحظ في البيانات الجديدة"},
    "sec6_posterior_subheader": {"en": "Posterior Beliefs (After Update)", "ar": "المعتقدات اللاحقة (بعد التحديث)"},
    "sec6_posterior_dist_template": {"en": "The posterior distribution is $Beta(\\alpha_0 + k, \\beta_0 + n - k) = Beta({posterior_alpha:.1f}, {posterior_beta:.1f})$",
                                     "ar": "التوزيع اللاحق هو $Beta(\\alpha_0 + k, \\beta_0 + n - k) = Beta({posterior_alpha:.1f}, {posterior_beta:.1f})$"},
    "sec6_posterior_mean_label": {"en": "Posterior Mean", "ar": "المتوسط اللاحق"},
    "sec6_posterior_ci_label": {"en": "95% Credible Interval (Posterior)", "ar": "فترة الموثوقية ٩٥٪ (اللاحقة)"},
    "sec6_target_width_label": {"en": "Target Credible Interval Width for Stopping", "ar": "عرض فترة الموثوقية المستهدف للإيقاف"},
    "sec6_target_met_success_template": {"en": "Target precision met! Current width ({current_width:.3f}) ≤ Target width ({target_width:.3f}).",
                                         "ar": "تم تحقيق الدقة المستهدفة! العرض الحالي ({current_width:.3f}) ≤ العرض المستهدف ({target_width:.3f})."},
    "sec6_target_not_met_warning_template": {"en": "Target precision not yet met. Current width ({current_width:.3f}) > Target width ({target_width:.3f}). Consider more samples.",
                                             "ar": "لم يتم تحقيق الدقة المستهدفة بعد. العرض الحالي ({current_width:.3f}) > العرض المستهدف ({target_width:.3f}). ضع في اعتبارك المزيد من العينات."},
    "sec6_plot1_title_base": {"en": "Prior and Posterior Distributions of Satisfaction Rate", "ar": "التوزيعات المسبقة واللاحقة لمعدل الرضا"},
    "sec6_plot_xlabel_base": {"en": "Satisfaction Rate (θ)", "ar": "معدل الرضا (θ)"},
    "sec6_plot_ylabel_base": {"en": "Density", "ar": "الكثافة"},
    "sec6_plot_prior_legend_base": {"en": "Prior", "ar": "المسبق"},
    "sec6_plot_posterior_legend_base": {"en": "Posterior", "ar": "اللاحق"},
    "sec6_discounting_subheader": {"en": "Conceptual Illustration: Impact of Discounting Older Data",
                                   "ar": "توضيح مفاهيمي: تأثير خصم البيانات القديمة"},
    "sec6_discounting_md": {"en": """
This illustrates how a discount factor might change the influence of 'old' posterior data when it's used to form a prior for a 'new' period.
Assume the 'Posterior' calculated above is now 'Old Data' from a previous period.
We want to form a new prior for the upcoming period.
An 'Initial Prior' (e.g., $Beta(1,1)$) represents a baseline, less informative belief.
    """, "ar": """
يوضح هذا كيف يمكن لعامل الخصم أن يغير تأثير بيانات "البيانات القديمة" اللاحقة عند استخدامها لتشكيل توزيع مسبق لفترة "جديدة".
افترض أن "التوزيع اللاحق" المحسوب أعلاه هو الآن "بيانات قديمة" من فترة سابقة.
نريد تشكيل توزيع مسبق جديد للفترة القادمة.
يمثل "التوزيع المسبق الأولي" (على سبيل المثال، $Beta(1,1)$) اعتقادًا أساسيًا أقل إفادة."""},
    "sec6_discount_factor_label": {"en": "Discount Factor (δ) for Old Data", "ar": "عامل الخصم (δ) للبيانات القديمة"},
    "sec6_discount_factor_help": {"en": "Controls weight of old data. 1.0 = full weight, 0.0 = discard old data, rely only on initial prior.",
                                  "ar": "يتحكم في وزن البيانات القديمة. ١.٠ = وزن كامل، ٠.٠ = تجاهل البيانات القديمة، والاعتماد فقط على التوزيع المسبق الأولي."},
    "sec6_init_prior_alpha_label": {"en": "Initial Prior Alpha (for new period if discounting heavily)", "ar": "ألفا المسبقة الأولية (لفترة جديدة إذا كان الخصم كبيرًا)"},
    "sec6_init_prior_beta_label": {"en": "Initial Prior Beta (for new period if discounting heavily)", "ar": "بيتا المسبقة الأولية (لفترة جديدة إذا كان الخصم كبيرًا)"},
    "sec6_new_prior_template": {"en": "New Prior for Next Period: $Beta({new_prior_alpha:.2f}, {new_prior_beta:.2f})$",
                              "ar": "التوزيع المسبق الجديد للفترة التالية: $Beta({new_prior_alpha:.2f}, {new_prior_beta:.2f})$"},
    "sec6_new_prior_mean_label": {"en": "Mean of New Prior", "ar": "متوسط التوزيع المسبق الجديد"},
    "sec6_plot2_title_base": {"en": "Forming a New Prior with Discounting", "ar": "تشكيل توزيع مسبق جديد مع الخصم"},
    "sec6_plot2_old_posterior_legend_base": {"en": "Old Posterior (Data from T-1)", "ar": "التوزيع اللاحق القديم (بيانات من T-1)"},
    "sec6_plot2_fixed_prior_legend_base": {"en": "Fixed Initial Prior", "ar": "التوزيع المسبق الأولي الثابت"},
    "sec6_plot2_new_prior_legend_base": {"en": "New Prior", "ar": "التوزيع المسبق الجديد"},

    # Section 7
    "sec7_title": {"en": "7. Conclusion", "ar": "٧. الخلاصة"},
    "sec7_conclusion_md": {"en": """
The proposed Bayesian adaptive estimation framework offers a sophisticated, flexible, and efficient approach to analyzing pilgrim satisfaction surveys. By iteratively updating beliefs and dynamically adjusting sampling efforts, this methodology promises more precise and timely insights, enabling better-informed decision-making for enhancing the Hajj experience.

While it introduces new concepts and requires careful implementation, the long-term benefits—including optimized resource use and a deeper understanding of satisfaction dynamics—are substantial. This proposal advocates for a phased implementation, starting with core metrics and gradually building complexity and scope.

We recommend proceeding with a pilot project to demonstrate the practical benefits and refine the operational aspects of this advanced analytical approach.
    """, "ar": """
يقدم إطار تقدير Bayesian التكيفي المقترح نهجًا متطورًا ومرنًا وفعالًا لتحليل استطلاعات رضا الحجاج. من خلال تحديث المعتقدات بشكل متكرر وتعديل جهود أخذ العينات ديناميكيًا، تعد هذه المنهجية برؤى أكثر دقة وفي الوقت المناسب، مما يتيح اتخاذ قرارات أفضل استنارة لتعزيز تجربة الحج.

في حين أنه يقدم مفاهيم جديدة ويتطلب تنفيذًا دقيقًا، فإن الفوائد طويلة الأجل - بما في ذلك الاستخدام الأمثل للموارد والفهم الأعمق لديناميكيات الرضا - كبيرة. يدعو هذا المقترح إلى تنفيذ مرحلي، بدءًا من المقاييس الأساسية وبناء التعقيد والنطاق تدريجيًا.

نوصي بالمضي قدمًا في مشروع تجريبي لإثبات الفوائد العملية وتحسين الجوانب التشغيلية لهذا النهج التحليلي المتقدم."""}
}

print("LANGUAGES dictionary defined successfully.")
