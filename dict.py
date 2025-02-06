from __future__ import annotations
import asyncio
import logging
from typing import List
from langsmith import traceable
from determination_agent import WordDetermination, get_determination
from determination_validator import vet_association_candidates
from phrase_extractor import split_segment
from cache import get_cached_associations, add_segment_to_cache
from db import record_determination

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@traceable(
    run_type="chain",
    name="Correct Words in Segment",
    project_name="Correct Words in Segment"
)
async def correct_words_in_segment(ref: str, segment: str) -> List[WordDetermination]:
    """

    :param ref: Sefaria Ref in normal form
    :param segment: The full text of the segment
    :return:
    """
    words = split_segment(segment)

    # Create state objects
    # I'm assuming that state_objects will be mutated in place
    state_objects = [
        {"ref": ref, "word": word, "segment": segment, "cached_associations": get_cached_associations(word), "selected_association": None, "determination": None}
        for word in words
    ]

    # Vet those that have possible associations.
    states_with_previous_associations = [state for state in state_objects if state["cached_associations"]]
    tasks = [asyncio.create_task(vet_association_candidates(state)) for state in states_with_previous_associations]
    await asyncio.gather(*tasks)

    # For words that haven't yet been resolved, call the model to determine the best entries
    states_without_resolution = [state for state in state_objects if not state["selected_association"]]
    tasks = [asyncio.create_task(get_determination(state)) for state in states_without_resolution]
    await asyncio.gather(*tasks)

    # Record the words and associations in the cache and in the DB.
    for state in state_objects:
        add_segment_to_cache(state)
        record_determination(state)


if __name__ == "__main__":
    # ref="Chullin 60a:2"
    # segment = '''אמר ליה קיסר לרבי יהושע בן חנניה בעינא דאיצבית ליה נהמא לאלהיכו אמר ליה לא מצית אמאי נפישי חילוותיה א"ל איברא אמר ליה פוק צבית לגידא דרביתא דרויחא עלמא טרח שיתא ירחי קייטא אתא זיקא כנשיה לימא'''

    #ref = "Jerusalem Talmud Demai 2.3.3"
    #segment = '''הָא מִכְּלָל דּוּ מוֹדֵי עַל קַמָּייָתָא. לֵית הָדָא פְלִיגָא עַל דְּרִבִּי יוֹנָה דְּרִבִּי יוֹנָה אָמַר חֲבֵירִין אֵינָן חֲשׁוּדִין לֹא לֶאֱכֹל וְלֹא לְהַאֲכִיל שֶׁלֹּא יֵלֵךְ וִיטַּמֵּא גוּפוֹ וְיָבֹא וִיטַמֵּא טָהֳרוֹת. וַאֲפִילוּ עַל דְּרִבִּי יוֹסֵי לֵית הָדָא פְלִיגָא. דְּרִבִּי יוֹסֵי אָמַר חֲבֵירִין חֲשׁוּדִין לוֹכַל וְאֵין חֲשׁוּדִין לְהַאֲכִיל. תַּמָּן לְטָהֳרוֹת אֲבָל הָכָא לְמַעְשְׂרוֹת. הַנֶּאֱמָן עַל הַטָּהֳרוֹת נֶאֱמָן עַל הַמַּעְשְׂרוֹת.'''

    #segment = ''' חַד בַּר נַשׁ הֲוָה קָאִים רְדִי. פָּֽסְקָת תּוֹרָתֵיהּ קוֹמוֹי. הֲװָת פָֽרְיָא וְהוּא פָרֵי פָֽרְיָא וְהוּא פָרֵי עַד דְּאִשְׁתְּכָח יְהִיב בְּבָבֶל. אָֽמְרוּ לֵיהּ אֵימַת נָֽפְקָת. אָמַר לוֹן יוֹמָא דֵין. אָֽמְרִין בְּהֵיידָא אָתִיתָא. אָמַר לוֹן בְּדָא. אָמַר לוֹן אִיתָא חָמֵי לֹן. נְפַק בְּעֵי מֵיחְמַייָא לוֹן וְלָא חַכִּים בְּהֵיידָא'''
    #ref = "Jerusalem Talmud Maaser Sheni 5:2:6"

    segment = ''' לִיבְטֵּיל — וַהֲלֹא לֹא נִבְרָא הָעוֹלָם אֶלָּא לִפְרִיָּה וּרְבִיָּה, שֶׁנֶּאֱמַר: ״לֹא תֹהוּ בְרָאָהּ לָשֶׁבֶת יְצָרָהּ״. אֶלָּא מִפְּנֵי תִּיקּוּן הָעוֹלָם, כּוֹפִין אֶת רַבּוֹ וְעוֹשֶׂה אוֹתוֹ בֶּן חוֹרִין, וְכוֹתֵב לוֹ שְׁטָר עַל חֲצִי דָּמָיו. וְחָזְרוּ בֵּית הִלֵּל לְהוֹרוֹת כְּדִבְרֵי בֵּית שַׁמַּאי.'''
    ref = "Chagigah 2b:2"

    determinations = asyncio.run(correct_words_in_segment(ref, segment))
    print(determinations)

