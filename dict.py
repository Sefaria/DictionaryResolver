from __future__ import annotations
import asyncio
import logging
from typing import List
from determination_agent import get_determination
from determination_validator import vet_association_candidates
from phrase_extractor import split_segment
from cache import get_cached_associations, add_segment_to_cache, add_empty_association_to_cache, clear_cache
from db import record_determination, record_empty_determination, clear_wordforms
from log import log, clear_log
import django
django.setup()
from sefaria.model import Ref, TextChunk

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def reset_system():
    """
    Reset the system by clearing the cache and the wordforms.
    This is useful for testing.  Once this hits production, probably not the best idea.
    :return:
    """
    logger.warning("Clearing Cache")
    clear_cache()
    logger.warning("Clearing Wordforms")
    clear_wordforms()
    logger.warning("Clearing Logs")
    clear_log()
    logger.warning("System Cleared")


async def correct_words_in_segment(ref: str, segment: str) -> List[dict]:
    """

    :param ref: Sefaria Ref in normal form
    :param segment: The full text of the segment
    :return:
    """
    logger.info(f"Splitting words for {ref}")
    words = split_segment(segment)

    # Create state objects
    # I'm assuming that state_objects will be mutated in place
    state_objects = [
        {"ref": ref, "word": word, "segment": segment, "cached_associations": get_cached_associations(word), "selected_association": None, "determination": None}
        for word in words
    ]

    for state in state_objects:
        log("Begin", state)

    # Vet those that have possible associations.
    logger.info(f"Vetting Association Candidates for {ref}")
    states_with_previous_associations = [state for state in state_objects if state["cached_associations"]]
    tasks = [asyncio.create_task(vet_association_candidates(state)) for state in states_with_previous_associations]
    await asyncio.gather(*tasks)

    # For words that haven't yet been resolved, call the model to determine the best entries
    logger.info(f"Getting Determinations for {ref}")
    states_without_resolution = [state for state in state_objects if state["selected_association"] is None]
    tasks = [asyncio.create_task(get_determination(state)) for state in states_without_resolution]
    await asyncio.gather(*tasks)

    # Record the words and associations in the cache and in the DB.
    for state in state_objects:
        if not state["selected_association"]:
            log("No Association Found", state)
            record_empty_determination(state)
            add_empty_association_to_cache(state)
        else:
            add_segment_to_cache(state)
            record_determination(state)

    return state_objects


if __name__ == "__main__":
    # ref="Chullin 60a:2"
    # segment = '''אמר ליה קיסר לרבי יהושע בן חנניה בעינא דאיצבית ליה נהמא לאלהיכו אמר ליה לא מצית אמאי נפישי חילוותיה א"ל איברא אמר ליה פוק צבית לגידא דרביתא דרויחא עלמא טרח שיתא ירחי קייטא אתא זיקא כנשיה לימא'''

    #ref = "Jerusalem Talmud Demai 2.3.3"
    #segment = '''הָא מִכְּלָל דּוּ מוֹדֵי עַל קַמָּייָתָא. לֵית הָדָא פְלִיגָא עַל דְּרִבִּי יוֹנָה דְּרִבִּי יוֹנָה אָמַר חֲבֵירִין אֵינָן חֲשׁוּדִין לֹא לֶאֱכֹל וְלֹא לְהַאֲכִיל שֶׁלֹּא יֵלֵךְ וִיטַּמֵּא גוּפוֹ וְיָבֹא וִיטַמֵּא טָהֳרוֹת. וַאֲפִילוּ עַל דְּרִבִּי יוֹסֵי לֵית הָדָא פְלִיגָא. דְּרִבִּי יוֹסֵי אָמַר חֲבֵירִין חֲשׁוּדִין לוֹכַל וְאֵין חֲשׁוּדִין לְהַאֲכִיל. תַּמָּן לְטָהֳרוֹת אֲבָל הָכָא לְמַעְשְׂרוֹת. הַנֶּאֱמָן עַל הַטָּהֳרוֹת נֶאֱמָן עַל הַמַּעְשְׂרוֹת.'''

    #segment = ''' חַד בַּר נַשׁ הֲוָה קָאִים רְדִי. פָּֽסְקָת תּוֹרָתֵיהּ קוֹמוֹי. הֲװָת פָֽרְיָא וְהוּא פָרֵי פָֽרְיָא וְהוּא פָרֵי עַד דְּאִשְׁתְּכָח יְהִיב בְּבָבֶל. אָֽמְרוּ לֵיהּ אֵימַת נָֽפְקָת. אָמַר לוֹן יוֹמָא דֵין. אָֽמְרִין בְּהֵיידָא אָתִיתָא. אָמַר לוֹן בְּדָא. אָמַר לוֹן אִיתָא חָמֵי לֹן. נְפַק בְּעֵי מֵיחְמַייָא לוֹן וְלָא חַכִּים בְּהֵיידָא'''
    #ref = "Jerusalem Talmud Maaser Sheni 5:2:6"

    #segment = ''' לִיבְטֵּיל — וַהֲלֹא לֹא נִבְרָא הָעוֹלָם אֶלָּא לִפְרִיָּה וּרְבִיָּה, שֶׁנֶּאֱמַר: ״לֹא תֹהוּ בְרָאָהּ לָשֶׁבֶת יְצָרָהּ״. אֶלָּא מִפְּנֵי תִּיקּוּן הָעוֹלָם, כּוֹפִין אֶת רַבּוֹ וְעוֹשֶׂה אוֹתוֹ בֶּן חוֹרִין, וְכוֹתֵב לוֹ שְׁטָר עַל חֲצִי דָּמָיו. וְחָזְרוּ בֵּית הִלֵּל לְהוֹרוֹת כְּדִבְרֵי בֵּית שַׁמַּאי.'''
    #ref = "Chagigah 2b:2"


    # determinations = asyncio.run(correct_words_in_segment(ref, segment))

    reset_system()
    page = Ref("Sanhedrin 63a")
    segments = page.all_segment_refs()
    for segment in segments:
        logging.warning(f"Processing {segment.normal()}")
        segment_text = TextChunk.remove_html_and_make_presentable(segment.text('he', vtitle="William Davidson Edition - Vocalized Aramaic").text)
        determinations = asyncio.run(correct_words_in_segment(segment.normal(), segment_text))
