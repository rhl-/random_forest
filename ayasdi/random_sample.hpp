#pragma once
/****************************************************************************
 * Original Author Keith Schwartz @ Stanford
 * Modified by Ryan H. Lewis 
 * License BSD-3 
 * 
 * Implementation of a random sampling algorithm on streams.  The algorithm
 * takes as input a stream of values (represented by a range of input
 * iterators), as well as an output range, then fills the output range with
 * elements sampled randomly from the input stream with uniform probability.
 * The algorithm need not know in advance the number of elements in the
 * stream.
 *
 * Internally, the algorithm works as follows.  Suppose that we are to pick
 * k elements at random.  Initially, the algorithm guesses that it will pick
 * the first k elements from the stream.  From that point forward, upon
 * seeing a new element, the algorithm chooses a random number in the range
 * [0, n], where n is the number of elements encountered so far, including
 * this new one.  If the value is in the range [0, k), then the element at
 * that position in the output is overwritten with the newly-sampled value.
 * Otherwise, the element is ignored.
 *
 * We can show that this chooses each element with uniform probability by
 * induction on the stream length.  This induction is valid because it does
 * not consider the length of the stream during its decision-making, and so
 * the behavior of the algorithm on a stream of n + 1 elements is the same
 * as the behavior of the algorithm on a stream of n elements, plus one
 * extra iteration of the loop.
 *
 * As a base case, if the length of the stream is k or less, then the
 * algorithm will pick each element, and so each element is correctly chosen
 * with uniform probability (since every element must be picked).
 *
 * For the inductive step, assume that for a stream of n elements, each
 * element is correctly picked with probability k / n and consider the
 * next iteration of the algorithm.  The probability that the algorithm
 * will choose to store the next value that comes from the stream is the
 * probability that a number in the range [0, n + 1] is in the range
 * [0, k).  This has probability k / (n + 1).  Moreover, consider the
 * probability that any element from the first n elements of the stream
 * is chosen.  By the inductive hypothesis, each element has probability
 * k / n of being chosen.  The probability that an element chosen this way
 * survives after this iteration is then the probability that the randomly-
 * chosen number is not equal to the slot in which that element is stored,
 * and since all slots are uniform this is n / (n + 1), since only one
 * choice can evict this element.  Thus the total probability that the
 * element is chosen is (k / n) * (n / (n + 1)) = k / (n + 1), and we
 * have that each element in the series has probability k / (n + 1) of
 * being chosen, which is a uniform distribution.
 *
 * The implementation provided here by default uses rand to generate random
 * numbers, but a custom random generator may be used instead.
 */
#include <cstdlib> // For rand
namespace ayasdi{
/**
 * Function: RandomSample(InputIterator in_begin, InputIterator in_end,
 *                        RandomAccessIterator out_begin, RandomAccessIterator out_end);
 * -------------------------------------------------------------------------
 * Populates the output range [out_begin, out_end) with a uniform random
 * sample of the elements in the range [in_begin, in_end).  Internally, this
 * function uses rand to generate random numbers.  If the input range does
 * not contain enough elements, then only some of the values will be filled
 * in and the algorithm will return an iterator to the last element written.
 * If at least out_end - out_begin elements were written, the return value
 * is out_end.
 */
template <typename InputIterator, typename RandomAccessIterator, typename RandomGenerator>
RandomAccessIterator random_sample(InputIterator in_begin, InputIterator in_end,
                                   RandomAccessIterator out_begin, RandomAccessIterator out_end, 
                                   RandomGenerator rng){
  /* Try reading in out_end - out_begin elements, aborting early if they can't
   * be read.
   */
  RandomAccessIterator itr = out_begin;
  for (; itr != out_end && in_begin != in_end; ++itr, ++in_begin){ *itr = *in_begin; }

  /* If we ran out of elements early, report that.  We can detect this by
   * checking whether our advancing iterator hit the end of the output
   * range.
   */
  if (itr != out_end){
    return itr;
  }

  /* For simplicity, cache the number of elements in the output range. */
  const size_t numOutputSlots = out_end - out_begin;

  /* Now apply the main algorithm by reading elements and deciding whether to
   * randomly evict an element or to skip it.
   */
  for (size_t count = numOutputSlots; in_begin != in_end; ++in_begin, ++count) {
    size_t index = rng() % (count + 1);
    if (index < numOutputSlots){
      out_begin[index] = *in_begin;
    }
  }

  /* Report that we read everything in by handing back the end of the output
   * range.
   */
  return out_end;
}


/**
 * Function: RandomSample(InputIterator in_begin, InputIterator in_end,
 *                        RandomAccessIterator out_begin, RandomAccessIterator out_end,
 *                        RandomGenerator rng);
 * -------------------------------------------------------------------------
 * Populates the output range [out_begin, out_end) with a uniform random
 * sample of the elements in the range [in_begin, in_end).  Internally, this
 * function uses the generator rng to generate random numbers.  If the input 
 * range does not contain enough elements, then only some of the values will
 * be filled in and the algorithm will return an iterator to the last element 
 * written.  If at least out_end - out_begin elements were written, the return 
 * value is out_end.
 */
template <typename InputIterator, typename RandomAccessIterator>
RandomAccessIterator random_sample(InputIterator in_begin, InputIterator in_end,
                            RandomAccessIterator out_begin, RandomAccessIterator out_end) {
  return random_sample(in_begin, in_end, out_begin, out_end, std::rand);
}
} //namespace ayasdi
